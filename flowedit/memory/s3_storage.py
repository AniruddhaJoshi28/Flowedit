"""
S3-Only Spelling Store for FlowEdit Pronunciation Corrections.

Maintains an in-memory dictionary of phonetic spelling corrections indexed by
(word, sense), and synchronizes directly with Amazon S3 / S3-compatible buckets.
No local disk files are created.
"""

import os
import re
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

from flowedit.alignment.homograph_resolver import HomographContextResolver
from flowedit.utils.env import load_flowedit_env

# Ensure .env is loaded
load_flowedit_env()

logger = logging.getLogger(__name__)


class S3SpellingStore:
    """In-memory dictionary with direct S3 persistence (no local disk files)."""

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        prefix: str = "corrections/",
        region_name: str = "us-east-1",
    ):
        self.bucket_name = (
            bucket_name
            or os.environ.get("FLOWEDIT_S3_BUCKET")
            or os.environ.get("S3_BUCKET_NAME")
            or os.environ.get("AWS_S3_BUCKET")
            or "flowedit-bucket"
        ).strip()
        self.endpoint_url = (
            endpoint_url
            or os.environ.get("FLOWEDIT_S3_URL")
            or os.environ.get("S3_ENDPOINT_URL", "")
        ).strip() or None
        self.prefix = (
            prefix
            or os.environ.get("FLOWEDIT_S3_PREFIX", "corrections/")
        ).strip()
        if not self.prefix.endswith("/"):
            self.prefix += "/"
        self.region_name = (
            region_name
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        ).strip()

        # In-memory dictionary: { word_lower: { "word": ..., "default_spell": ..., "senses": [...] } }
        self._dictionary: Dict[str, Dict[str, Any]] = {}
        self.resolver = HomographContextResolver()
        self._s3_client = None

        # Attempt to load existing dictionary if S3 is configured
        if self.bucket_name:
            try:
                self.load_from_s3()
            except Exception as e:
                logger.warning(f"Could not load initial dictionary from S3: {e}")

    def _get_s3_client(self):
        """Lazy-initialize boto3 S3 client."""
        if self._s3_client is not None:
            return self._s3_client
        try:
            import boto3
            from botocore.config import Config

            load_flowedit_env()

            cfg = Config(
                region_name=self.region_name,
                retries={"max_attempts": 3, "mode": "standard"},
            )
            kwargs = {"config": cfg}
            if self.endpoint_url:
                kwargs["endpoint_url"] = self.endpoint_url

            # Use env AWS credentials if present
            if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
                kwargs["aws_access_key_id"] = os.environ.get("AWS_ACCESS_KEY_ID")
                kwargs["aws_secret_access_key"] = os.environ.get("AWS_SECRET_ACCESS_KEY")
            if os.environ.get("AWS_SESSION_TOKEN"):
                kwargs["aws_session_token"] = os.environ.get("AWS_SESSION_TOKEN")

            profile_name = os.environ.get("AWS_PROFILE")
            session = boto3.Session(profile_name=profile_name) if profile_name else boto3.Session()
            self._s3_client = session.client("s3", **kwargs)
            return self._s3_client
        except Exception as e:
            logger.warning(f"Failed to create boto3 S3 client: {e}")
            return None

    def configure_s3(
        self,
        bucket_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        prefix: Optional[str] = None,
        region_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Configure or update S3 bucket connection details."""
        if bucket_name is not None:
            self.bucket_name = bucket_name.strip()
        if endpoint_url is not None:
            self.endpoint_url = endpoint_url.strip() or None
        if prefix is not None:
            self.prefix = prefix.strip()
            if not self.prefix.endswith("/"):
                self.prefix += "/"
        if region_name is not None:
            self.region_name = region_name.strip()

        # Reset client to reconfigure
        self._s3_client = None
        logger.info(f"[S3 Store] Configured S3: bucket='{self.bucket_name}', prefix='{self.prefix}'")

        # If bucket is configured and we have memory entries, sync to S3
        sync_res = {}
        if self.bucket_name:
            sync_res = self.sync_to_s3()

        return {
            "bucket_name": self.bucket_name,
            "endpoint_url": self.endpoint_url,
            "prefix": self.prefix,
            "region_name": self.region_name,
            "sync_result": sync_res,
        }

    def add_correction(
        self,
        word: str,
        spell_as: str,
        carrier_text: str = "",
        sense_id: Optional[str] = None,
        language: str = "en",
    ) -> Dict[str, Any]:
        """Register a phonetic spelling correction for (word, sense) and sync to S3.
        
        Args:
            word: Target word (e.g. 'read')
            spell_as: Replacement spelling (e.g. 'red')
            carrier_text: Optional sentence context (e.g. 'she read the book')
            sense_id: Optional sense ID override (if None, classified by HomographContextResolver)
            language: Language code
            
        Returns:
            Dict containing registration details and S3 sync status.
        """
        word_clean = word.strip()
        word_key = word_clean.lower()
        spell_clean = spell_as.strip()

        # Classify sense using the shared HomographContextResolver
        profile = self.resolver.classify_sense(carrier_text or word_clean, word_clean)
        resolved_sense_id = sense_id.strip() if sense_id else profile.sense_id
        sense_display = profile.display_name if not sense_id else f"Sense: {resolved_sense_id}"

        now_iso = datetime.now(timezone.utc).isoformat()

        sense_entry = {
            "sense_id": resolved_sense_id,
            "display_name": sense_display,
            "spell_as": spell_clean,
            "carrier_text": carrier_text,
            "context_window": profile.context_window,
            "left_tokens": profile.left_tokens,
            "right_tokens": profile.right_tokens,
            "immediate_left": profile.immediate_left,
            "immediate_right": profile.immediate_right,
            "context_clues": profile.keywords,
            "language": language,
            "updated_at": now_iso,
        }

        if word_key not in self._dictionary:
            self._dictionary[word_key] = {
                "word": word_clean,
                "default_spell": spell_clean,
                "senses": [sense_entry],
                "created_at": now_iso,
            }
        else:
            word_dict = self._dictionary[word_key]
            word_dict["word"] = word_clean
            word_dict["default_spell"] = spell_clean

            # Check if this sense_id already exists in senses list
            existing_idx = next(
                (i for i, s in enumerate(word_dict["senses"]) if s.get("sense_id") == resolved_sense_id),
                None,
            )
            if existing_idx is not None:
                word_dict["senses"][existing_idx] = sense_entry
            else:
                word_dict["senses"].append(sense_entry)

        logger.info(
            f"[S3 Store] Registered spelling: '{word_clean}' -> '{spell_clean}' "
            f"(sense='{resolved_sense_id}', carrier='{carrier_text}')"
        )

        # Sync to S3 bucket
        s3_res = self.sync_to_s3()

        return {
            "success": True,
            "mode": "spell",
            "word": word_clean,
            "spell_as": spell_clean,
            "sense_id": resolved_sense_id,
            "sense_display": sense_display,
            "carrier_text": carrier_text,
            "s3_synced": s3_res.get("synced", False),
            "s3_status": s3_res.get("status", "pending_s3_link"),
            "s3_uri": s3_res.get("s3_uri"),
            "total_entries": len(self._dictionary),
        }

    def resolve_correction(self, text: str, word: str) -> Optional[str]:
        """Deterministically resolve the correct phonetic spelling for word in text context.
        
        Evaluates context compatibility against registered senses. If context does
        not match the registered sense(s), returns None (preserves original word).
        
        Args:
            text: Full carrier sentence
            word: Target word
            
        Returns:
            The replacement spelling string (e.g. 'red') or None if not registered/incompatible.
        """
        word_key = word.strip().lower()
        if word_key not in self._dictionary:
            return None

        word_dict = self._dictionary[word_key]
        senses = word_dict.get("senses", [])

        if not senses:
            return word_dict.get("default_spell")

        matched_sense = self.resolver.match_sense(text, word_key, senses)
        if matched_sense and "spell_as" in matched_sense:
            return matched_sense["spell_as"]

        return None

    def apply_corrections_to_text(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Scan text and deterministically replace any matching dictionary words with phonetic spelling.
        
        Uses whole-word boundary matching with context-sensitive sense disambiguation.
        
        Args:
            text: Original carrier sentence
            
        Returns:
            Tuple of (refined_text, list_of_applied_corrections)
        """
        if not self._dictionary or not text:
            return text, []

        applied = []
        result_text = text

        for word_key, word_dict in self._dictionary.items():
            pattern = re.compile(r'\b' + re.escape(word_key) + r'\b', re.IGNORECASE)
            matches = list(pattern.finditer(result_text))
            if not matches:
                continue

            resolved_spell = self.resolve_correction(result_text, word_key)
            if resolved_spell and resolved_spell.strip().lower() != word_key.lower():
                # Perform whole-word substitution
                result_text = pattern.sub(resolved_spell.strip(), result_text)
                applied.append({
                    "word": word_dict.get("word", word_key),
                    "spell_as": resolved_spell.strip(),
                    "original_text": text,
                })
                logger.info(
                    f"[S3 Store] Applied deterministic phonetic respelling: "
                    f"'{word_dict.get('word', word_key)}' -> '{resolved_spell.strip()}'"
                )

        return result_text, applied

    def sync_to_s3(self) -> Dict[str, Any]:
        """Upload the current dictionary to S3 bucket (no local file writes)."""
        if not self.bucket_name:
            logger.info("[S3 Store] S3 bucket not configured yet. Dictionary held in memory. Will sync when link is provided.")
            return {
                "synced": False,
                "status": "pending_s3_link",
                "message": "S3 bucket link not configured yet. Held in memory.",
                "total_words": len(self._dictionary),
            }

        client = self._get_s3_client()
        if client is None:
            return {
                "synced": False,
                "status": "boto3_unavailable",
                "message": "Could not initialize AWS S3 client.",
                "total_words": len(self._dictionary),
            }

        key = f"{self.prefix}spelling_dictionary.json"
        try:
            payload = {
                "version": "1.0",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "total_words": len(self._dictionary),
                "words": self._dictionary,
            }
            body_bytes = json.dumps(payload, indent=2).encode("utf-8")

            client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=body_bytes,
                ContentType="application/json",
            )
            s3_uri = f"s3://{self.bucket_name}/{key}"
            logger.info(f"[S3 Store] ✓ Synchronized {len(self._dictionary)} words to S3: {s3_uri}")
            return {
                "synced": True,
                "status": "synced",
                "s3_uri": s3_uri,
                "bucket": self.bucket_name,
                "key": key,
                "total_words": len(self._dictionary),
            }
        except Exception as e:
            logger.error(f"[S3 Store] Error uploading to S3 ({self.bucket_name}/{key}): {e}")
            self._s3_client = None  # Reset client so next request re-loads fresh credentials
            return {
                "synced": False,
                "status": "upload_failed",
                "error": str(e),
                "total_words": len(self._dictionary),
            }

    def load_from_s3(self, replace: bool = False) -> bool:
        """Download dictionary from S3 bucket into memory.
        
        Args:
            replace: If True, clears existing in-memory entries before loading from S3.
        """
        if not self.bucket_name:
            return False

        client = self._get_s3_client()
        if client is None:
            return False

        key = f"{self.prefix}spelling_dictionary.json"
        try:
            resp = client.get_object(Bucket=self.bucket_name, Key=key)
            content = resp["Body"].read().decode("utf-8")
            data = json.loads(content)
            loaded_words = data.get("words", {})
            if isinstance(loaded_words, dict):
                if replace:
                    self._dictionary.clear()
                self._dictionary.update(loaded_words)
                logger.info(f"[S3 Store] ✓ Loaded {len(loaded_words)} words from {self.bucket_name}/{key}")
                return True
        except Exception as e:
            logger.info(f"[S3 Store] No existing dictionary in S3 ({self.bucket_name}/{key}): {e}")
        return False

    def get_s3_metadata(self) -> Dict[str, Any]:
        """Fetch remote S3 object metadata if accessible."""
        if not self.bucket_name:
            return {"exists": False, "reason": "no_bucket"}
        client = self._get_s3_client()
        if not client:
            return {"exists": False, "reason": "no_client"}
        key = f"{self.prefix}spelling_dictionary.json"
        try:
            head = client.head_object(Bucket=self.bucket_name, Key=key)
            last_mod = head.get("LastModified")
            iso_mod = last_mod.isoformat() if hasattr(last_mod, "isoformat") else str(last_mod)
            return {
                "exists": True,
                "bucket": self.bucket_name,
                "key": key,
                "size_bytes": head.get("ContentLength", 0),
                "last_modified": iso_mod,
                "etag": (head.get("ETag") or "").strip('"'),
                "content_type": head.get("ContentType", "application/json"),
            }
        except Exception as e:
            return {
                "exists": False,
                "bucket": self.bucket_name,
                "key": key,
                "error": str(e),
            }

    def get_entries_data(self, refresh: bool = False, word: Optional[str] = None) -> Dict[str, Any]:
        """Get dictionary entries along with live S3 status metadata.
        
        Args:
            refresh: If True, pulls fresh state from S3 before returning.
            word: Optional target word filter.
        """
        if refresh:
            self.load_from_s3(replace=True)

        meta = self.get_s3_metadata()
        key = f"{self.prefix}spelling_dictionary.json"

        if word:
            word_key = word.strip().lower()
            entry = self._dictionary.get(word_key)
            entries = [entry] if entry else []
        else:
            entries = list(self._dictionary.values())

        return {
            "bucket": self.bucket_name,
            "region": self.region_name,
            "prefix": self.prefix,
            "key": key,
            "s3_uri": f"s3://{self.bucket_name}/{key}" if self.bucket_name else None,
            "total_words": len(self._dictionary),
            "filtered_count": len(entries),
            "s3_metadata": meta,
            "entries": entries,
        }

    def delete_word(self, word: str) -> Tuple[bool, Dict[str, Any]]:
        """Delete a word from dictionary and sync updated state to S3.
        
        Returns:
            Tuple of (bool found, dict sync_result)
        """
        word_key = word.strip().lower()
        if word_key in self._dictionary:
            del self._dictionary[word_key]
            sync_res = self.sync_to_s3()
            return True, sync_res
        return False, {}

    def delete_sense(self, word: str, sense_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Delete a specific sense profile for a word and sync to S3.
        
        If no senses remain for the word, the word entry is removed.
        """
        word_key = word.strip().lower()
        if word_key not in self._dictionary:
            return False, {"error": f"Word '{word}' not found in dictionary."}

        word_data = self._dictionary[word_key]
        senses = word_data.get("senses", [])
        new_senses = [s for s in senses if s.get("sense_id") != sense_id]

        if len(new_senses) == len(senses):
            return False, {"error": f"Sense '{sense_id}' not found for word '{word}'."}

        if not new_senses:
            del self._dictionary[word_key]
        else:
            word_data["senses"] = new_senses
            # Update default spell to first remaining sense if any
            if new_senses and "spell_as" in new_senses[0]:
                word_data["default_spell"] = new_senses[0]["spell_as"]

        sync_res = self.sync_to_s3()
        return True, sync_res

    def list_corrections(self) -> List[Dict[str, Any]]:
        """Return all dictionary entries."""
        return list(self._dictionary.values())

    def clear_all(self, delete_remote_file: bool = False) -> Dict[str, Any]:
        """Clear all in-memory entries and update S3 bucket.
        
        Args:
            delete_remote_file: If True, completely deletes the object from S3.
                                If False, syncs an empty dictionary {} to S3.
        """
        self._dictionary.clear()
        if not self.bucket_name:
            return {"synced": False, "status": "no_bucket_configured", "total_words": 0}

        key = f"{self.prefix}spelling_dictionary.json"
        if delete_remote_file:
            client = self._get_s3_client()
            if client:
                try:
                    client.delete_object(Bucket=self.bucket_name, Key=key)
                    logger.info(f"[S3 Store] Deleted remote object s3://{self.bucket_name}/{key}")
                    return {
                        "synced": True,
                        "status": "remote_file_deleted",
                        "bucket": self.bucket_name,
                        "key": key,
                        "total_words": 0,
                    }
                except Exception as e:
                    logger.error(f"[S3 Store] Error deleting remote object s3://{self.bucket_name}/{key}: {e}")
                    return {"synced": False, "status": "delete_failed", "error": str(e), "total_words": 0}

        return self.sync_to_s3()

    def clear(self) -> None:
        """Clear all in-memory entries and sync to S3."""
        self.clear_all(delete_remote_file=False)


# Global singleton store instance
s3_spelling_store = S3SpellingStore()
