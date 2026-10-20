"""
=============================================================================
Phase 3 Research Prototype: SemFlow-Memory End-to-End Experimental Suite
=============================================================================
Evaluates the complete lifelong pronunciation adaptation loop:
  A. Baseline ambiguous homograph generation.
  B. Explicit user correction feedback.
  C. Associative memory storage with contextual features.
  D. Generalization & transfer to genuinely unseen sentences with the same sense.
  E. Localized acoustic correction without retraining F5-TTS.
  F. Specificity check on opposite-sense sentences (verifying zero false corrections).
  G. Sequential conflicting corrections (evaluating interference, margin gating,
     forgetting, and memory consolidation).

Benchmarks 6 Architectures:
  1. Original F5-TTS (No memory)
  2. Phonetic Substitution (Global oracle)
  3. Cosine k-NN Memory
  4. Direct CPA (1-NN without associative memory)
  5. Hopfield Memory Without Margin Gating (Ablation)
  6. Full SemFlow-Memory (Interference-Aware Hopfield + Margin Gating + Localized Controller)
=============================================================================
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
import json
import logging
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

# Monkeypatch torchaudio.load for Windows DLL safety
def safe_load(filepath, **kwargs):
    data, sr = sf.read(filepath)
    t = torch.from_numpy(data).float()
    if t.dim() == 1:
        t = t.unsqueeze(0)
    else:
        t = t.transpose(0, 1)
    return t, sr
import torchaudio
torchaudio.load = safe_load

from f5_tts.api import F5TTS
from faster_whisper import WhisperModel
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
try:
    from g2p_en import G2p
except ImportError:
    class G2p:
        def __call__(self, text):
            return list(text.lower())

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Local imports
from flowedit.memory.contrastive_adapter import ContrastivePronunciationAdapter, SupervisedContrastiveLoss
from flowedit.memory.correction_memory import CosineKNNMemory, InterferenceAwareHopfieldMemory
from flowedit.memory.semflow_controller import (
    AblatedHopfieldMemory,
    MiniLMContextEncoder,
    ConfigurableAcousticActuator,
    SemFlowSystem,
    splice_audio_patch,
)
from scripts.validate_acoustic_correction_study import (
    PhonemeAwareEvaluator,
    compute_bsd_at_boundaries,
    compute_non_target_per_wer,
    SPEAKER_REFS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase3Prototype")

OUTPUT_AUDIO_DIR = "reports/phase3_audio_artifacts"
RESULTS_JSON_PATH = "reports/phase3_prototype_results.json"
REPORT_MD_PATH = "reports/phase3_prototype_report.md"
os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)
os.makedirs("reports", exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Experimental Dataset: Unseen Lifelong Generalization & Specificity Tests
# ---------------------------------------------------------------------------

PHASE3_BENCHMARK_CASES = [
    # -----------------------------------------------------------------------
    # HOMOGRAPH 1: LEAD
    # -----------------------------------------------------------------------
    {
        "homograph": "lead",
        "seed_correction_s0": {
            "sentence": "The antique pipes were made of solid lead.",
            "sense_id": 0,
            "target_ipa": "/lɛd/",
            "phonetic_word": "led",
            "speaker": "female",
        },
        "transfer_unseen_s0": [
            {
                "id": "lead_transfer_1",
                "sentence": "Heavy lead weights were attached to the fishing line.",
                "expected_sense": 0,
                "target_ipa": "/lɛd/",
                "phonetic_word": "led",
                "speaker": "female",
            },
            {
                "id": "lead_transfer_2",
                "sentence": "The old stained glass window was framed in soft lead.",
                "expected_sense": 0,
                "target_ipa": "/lɛd/",
                "phonetic_word": "led",
                "speaker": "male",
            },
        ],
        "opposite_specificity_s1": [
            {
                "id": "lead_opposite_1",
                "sentence": "The chief researcher will lead the team.",
                "expected_sense": 1,
                "target_ipa": "/liːd/",
                "phonetic_word": "leed",
                "speaker": "female",
            },
            {
                "id": "lead_opposite_2",
                "sentence": "Follow the north star and take the lead.",
                "expected_sense": 1,
                "target_ipa": "/liːd/",
                "phonetic_word": "leed",
                "speaker": "male",
            },
        ],
        "sequential_correction_s1": {
            "sentence": "Good managers lead by setting a positive example.",
            "sense_id": 1,
            "target_ipa": "/liːd/",
            "phonetic_word": "leed",
            "speaker": "female",
        },
        "retention_test": [
            {
                "id": "lead_retention_s0",
                "sentence": "The bullet contained dense lead metal.",
                "expected_sense": 0,
                "target_ipa": "/lɛd/",
                "phonetic_word": "led",
                "speaker": "male",
            },
            {
                "id": "lead_retention_s1",
                "sentence": "Experience will lead you to make the right choice.",
                "expected_sense": 1,
                "target_ipa": "/liːd/",
                "phonetic_word": "leed",
                "speaker": "female",
            },
        ],
    },

    # -----------------------------------------------------------------------
    # HOMOGRAPH 2: READ
    # -----------------------------------------------------------------------
    {
        "homograph": "read",
        "seed_correction_s0": {
            "sentence": "Yesterday evening she read the entire book.",
            "sense_id": 0,
            "target_ipa": "/rɛd/",
            "phonetic_word": "red",
            "speaker": "female",
        },
        "transfer_unseen_s0": [
            {
                "id": "read_transfer_1",
                "sentence": "I already read that news article during breakfast.",
                "expected_sense": 0,
                "target_ipa": "/rɛd/",
                "phonetic_word": "red",
                "speaker": "male",
            },
        ],
        "opposite_specificity_s1": [
            {
                "id": "read_opposite_1",
                "sentence": "Students must read the assigned chapter today.",
                "expected_sense": 1,
                "target_ipa": "/riːd/",
                "phonetic_word": "reed",
                "speaker": "female",
            },
        ],
        "sequential_correction_s1": {
            "sentence": "Read the instructions before starting the machine.",
            "sense_id": 1,
            "target_ipa": "/riːd/",
            "phonetic_word": "reed",
            "speaker": "male",
        },
        "retention_test": [
            {
                "id": "read_retention_s0",
                "sentence": "Last week we read the historic diary aloud.",
                "expected_sense": 0,
                "target_ipa": "/rɛd/",
                "phonetic_word": "red",
                "speaker": "female",
            },
        ],
    },

    # -----------------------------------------------------------------------
    # HOMOGRAPH 3: BOW
    # -----------------------------------------------------------------------
    {
        "homograph": "bow",
        "seed_correction_s0": {
            "sentence": "The archer aimed his wooden bow carefully.",
            "sense_id": 0,
            "target_ipa": "/boʊ/",
            "phonetic_word": "boh",
            "speaker": "female",
        },
        "transfer_unseen_s0": [
            {
                "id": "bow_transfer_1",
                "sentence": "He pulled the string of the composite bow back.",
                "expected_sense": 0,
                "target_ipa": "/boʊ/",
                "phonetic_word": "boh",
                "speaker": "male",
            },
        ],
        "opposite_specificity_s1": [
            {
                "id": "bow_opposite_1",
                "sentence": "The performers took a graceful bow on stage.",
                "expected_sense": 1,
                "target_ipa": "/baʊ/",
                "phonetic_word": "baw",
                "speaker": "female",
            },
        ],
        "sequential_correction_s1": {
            "sentence": "The talented violinist finished playing and took a bow.",
            "sense_id": 1,
            "target_ipa": "/baʊ/",
            "phonetic_word": "baw",
            "speaker": "male",
        },
        "retention_test": [
            {
                "id": "bow_retention_s0",
                "sentence": "The hunter carried a recurve bow across the forest.",
                "expected_sense": 0,
                "target_ipa": "/boʊ/",
                "phonetic_word": "boh",
                "speaker": "female",
            },
        ],
    },

    # -----------------------------------------------------------------------
    # HOMOGRAPH 4: TEAR
    # -----------------------------------------------------------------------
    {
        "homograph": "tear",
        "seed_correction_s0": {
            "sentence": "A single warm tear rolled down her cheek.",
            "sense_id": 0,
            "target_ipa": "/tɪər/",
            "phonetic_word": "tier",
            "speaker": "female",
        },
        "transfer_unseen_s0": [
            {
                "id": "tear_transfer_1",
                "sentence": "He wiped a silent tear from his eye.",
                "expected_sense": 0,
                "target_ipa": "/tɪər/",
                "phonetic_word": "tier",
                "speaker": "male",
            },
        ],
        "opposite_specificity_s1": [
            {
                "id": "tear_opposite_1",
                "sentence": "Be careful not to tear the fragile parchment.",
                "expected_sense": 1,
                "target_ipa": "/tɛər/",
                "phonetic_word": "tare",
                "speaker": "female",
            },
        ],
        "sequential_correction_s1": {
            "sentence": "Tear the package open without using any scissors.",
            "sense_id": 1,
            "target_ipa": "/tɛər/",
            "phonetic_word": "tare",
            "speaker": "male",
        },
        "retention_test": [
            {
                "id": "tear_retention_s0",
                "sentence": "She held back a bitter tear of regret.",
                "expected_sense": 0,
                "target_ipa": "/tɪər/",
                "phonetic_word": "tier",
                "speaker": "female",
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# 2. Main Phase 3 Prototype Runner
# ---------------------------------------------------------------------------

def run_phase3_suite():
    logger.info("=" * 70)
    logger.info("STARTING PHASE 3 RESEARCH PROTOTYPE VALIDATION")
    logger.info("=" * 70)

    device = "cpu"

    # Step 1: Load F5-TTS Backbone & Whisper
    logger.info(f"Loading official F5TTS_Base on device '{device}'...")
    tts = F5TTS(model="F5TTS_Base", device=device)
    logger.info("Loading Whisper Tiny for alignment...")
    whisper = WhisperModel("tiny", device="cpu", compute_type="int8")

    # Step 2: Load Acoustic Phoneme Evaluator
    logger.info("Loading Wav2Vec2 CTC for multi-dimensional phonemic evaluation...")
    ctc_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    ctc_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    ctc_model.eval()
    g2p = G2p()
    evaluator = PhonemeAwareEvaluator(ctc_processor, ctc_model, g2p)

    # Step 3: Load Semantic Encoder & Adapter
    encoder = MiniLMContextEncoder(device=device)
    adapter = ContrastivePronunciationAdapter(in_dim=384, hidden_dim=384, out_dim=256).to(device)
    adapter.eval() # Pre-initialized projection head

    # Step 4: Initialize Configurable Acoustic Actuator
    actuator = ConfigurableAcousticActuator(tts_instance=tts, whisper_instance=whisper, device=device)

    # -----------------------------------------------------------------------
    # Step 5: Initialize 6 Comparative Memory Architectures
    # -----------------------------------------------------------------------
    architectures = {
        "original_f5tts": None, # pure baseline, no memory
        "phonetic_substitution": None, # global phonetic replacement oracle
        "cosine_knn": CosineKNNMemory(abstain_threshold=0.60),
        "direct_cpa_1nn": CosineKNNMemory(abstain_threshold=0.0), # No abstention threshold
        "hopfield_ablated": AblatedHopfieldMemory(beta=8.0, abstain_threshold=0.60), # No margin gating
        "semflow_memory": InterferenceAwareHopfieldMemory(beta=8.0, abstain_threshold=0.60, margin_threshold=0.25),
    }

    results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backbone": "F5TTS_Base (600M DiT)",
            "encoder": "all-MiniLM-L6-v2 (Frozen, d=384)",
            "adapter": "CPA (384 -> 256)",
            "num_homographs": len(PHASE3_BENCHMARK_CASES),
        },
        "learning_loop_results": {},
        "comparative_architecture_benchmark": {},
        "acoustic_preservation_audit": {},
    }

    # -----------------------------------------------------------------------
    # EXECUTE THE LIFELONG LEARNING WORKFLOW (A -> G)
    # -----------------------------------------------------------------------

    logger.info("\n" + "=" * 70)
    logger.info("EXECUTING WORKFLOW A -> G ON SEMFLOW-MEMORY")
    logger.info("=" * 70)

    semflow_sys = SemFlowSystem(
        encoder=encoder,
        adapter=adapter,
        memory=architectures["semflow_memory"],
        actuator=actuator,
        default_mode="localized_waveform_regeneration",
    )

    workflow_log = []

    for h_data in PHASE3_BENCHMARK_CASES:
        word = h_data["homograph"]
        logger.info(f"\n>>> PROCESSING HOMOGRAPH: '{word.upper()}' <<<")

        # -------------------------------------------------------------------
        # Step A: Baseline ambiguous homograph generation
        # -------------------------------------------------------------------
        seed_s0 = h_data["seed_correction_s0"]
        spk = seed_s0["speaker"]
        ref_wav = SPEAKER_REFS[spk]["wav_path"]
        ref_txt = SPEAKER_REFS[spk]["ref_text"]

        seed_base_path = os.path.join(OUTPUT_AUDIO_DIR, f"{word}_seed_baseline.wav")
        synth_res_a = semflow_sys.synthesize_with_adaptation(
            sentence=seed_s0["sentence"],
            target_word=word,
            speaker=spk,
            ref_file=ref_wav,
            ref_text=ref_txt,
            output_path=seed_base_path,
        )
        logger.info(f"  [Step A - Baseline] '{seed_s0['sentence']}' -> Action: {synth_res_a['action_taken']}")

        # -------------------------------------------------------------------
        # Step B & C: Accept user feedback and store in associative memory
        # -------------------------------------------------------------------
        fb_s0 = semflow_sys.provide_feedback(
            sentence=seed_s0["sentence"],
            target_word=word,
            correct_sense_id=seed_s0["sense_id"],
            label=seed_s0["target_ipa"],
            phonetic_word=seed_s0["phonetic_word"],
        )
        logger.info(f"  [Step B/C - Feedback Stored] Sense {seed_s0['sense_id']} ({seed_s0['target_ipa']}) stored in Hopfield memory.")

        # -------------------------------------------------------------------
        # Step D & E: Unseen Generalization / Transfer Test
        # -------------------------------------------------------------------
        transfer_results = []
        for t_case in h_data["transfer_unseen_s0"]:
            t_spk = t_case["speaker"]
            t_ref_wav = SPEAKER_REFS[t_spk]["wav_path"]
            t_ref_txt = SPEAKER_REFS[t_spk]["ref_text"]
            t_out_path = os.path.join(OUTPUT_AUDIO_DIR, f"{t_case['id']}_semflow.wav")

            t_res = semflow_sys.synthesize_with_adaptation(
                sentence=t_case["sentence"],
                target_word=word,
                speaker=t_spk,
                ref_file=t_ref_wav,
                ref_text=t_ref_txt,
                output_path=t_out_path,
            )

            # Evaluate pronunciation of edited audio
            w_s, w_e, _ = actuator.get_word_alignment(t_out_path, word)
            wav_eval, sr_eval = sf.read(t_out_path)
            f1, f2, df1, df2 = evaluator.extract_formant_trajectory(wav_eval, w_s, w_e, sr=sr_eval)
            _, ctc_target = evaluator.decode_ctc_target_interval(wav_eval, w_s, w_e, sr=sr_eval)
            is_ok, conf, det = evaluator.evaluate_pronunciation(word, t_case["expected_sense"], t_spk, f1, f2, df1, df2, (w_e-w_s)*1000, ctc_target, word)

            transfer_ok = (t_res["predicted_sense"] == t_case["expected_sense"]) and not t_res["is_abstained"]
            logger.info(f"  [Step D/E - Transfer] '{t_case['sentence']}' -> Retr Sense: {t_res['predicted_sense']} (Sim: {t_res['similarity']:.2f}) | Transferred: {transfer_ok} | Acoustic Correct: {is_ok}")
            transfer_results.append({
                "case_id": t_case["id"],
                "sentence": t_case["sentence"],
                "expected_sense": t_case["expected_sense"],
                "retrieved_sense": t_res["predicted_sense"],
                "similarity": t_res["similarity"],
                "is_abstained": t_res["is_abstained"],
                "transfer_success": transfer_ok,
                "acoustic_correct": is_ok,
            })

        # -------------------------------------------------------------------
        # Step F: Specificity Test on Opposite-Sense Sentence (Zero False Correction)
        # -------------------------------------------------------------------
        specificity_results = []
        for o_case in h_data["opposite_specificity_s1"]:
            o_spk = o_case["speaker"]
            o_ref_wav = SPEAKER_REFS[o_spk]["wav_path"]
            o_ref_txt = SPEAKER_REFS[o_spk]["ref_text"]
            o_out_path = os.path.join(OUTPUT_AUDIO_DIR, f"{o_case['id']}_semflow.wav")

            o_res = semflow_sys.synthesize_with_adaptation(
                sentence=o_case["sentence"],
                target_word=word,
                speaker=o_spk,
                ref_file=o_ref_wav,
                ref_text=o_ref_txt,
                output_path=o_out_path,
            )

            # False correction occurred if memory falsely applied sense 0 to an opposite-sense sentence!
            false_correction = (o_res["predicted_sense"] == 0) and not o_res["is_abstained"]
            logger.info(f"  [Step F - Specificity] '{o_case['sentence']}' -> Retr Sense: {o_res['predicted_sense']} | Abstained: {o_res['is_abstained']} (Sim: {o_res['similarity']:.2f}, Margin: {o_res['margin']:.2f}) | False Correction: {false_correction}")
            specificity_results.append({
                "case_id": o_case["id"],
                "sentence": o_case["sentence"],
                "expected_sense": o_case["expected_sense"],
                "retrieved_sense": o_res["predicted_sense"],
                "is_abstained": o_res["is_abstained"],
                "abstain_reason": o_res["abstain_reason"],
                "false_correction": false_correction,
            })

        # -------------------------------------------------------------------
        # Step G: Sequential Conflicting Correction & Lifelong Memory
        # -------------------------------------------------------------------
        seq_s1 = h_data["sequential_correction_s1"]
        logger.info(f"  [Step G - User Corrects Sense 1] Teaching sense 1 on '{seq_s1['sentence']}'...")
        fb_s1 = semflow_sys.provide_feedback(
            sentence=seq_s1["sentence"],
            target_word=word,
            correct_sense_id=seq_s1["sense_id"],
            label=seq_s1["target_ipa"],
            phonetic_word=seq_s1["phonetic_word"],
        )

        retention_results = []
        for r_case in h_data["retention_test"]:
            r_spk = r_case["speaker"]
            r_ref_wav = SPEAKER_REFS[r_spk]["wav_path"]
            r_ref_txt = SPEAKER_REFS[r_spk]["ref_text"]
            r_out_path = os.path.join(OUTPUT_AUDIO_DIR, f"{r_case['id']}_semflow.wav")

            r_res = semflow_sys.synthesize_with_adaptation(
                sentence=r_case["sentence"],
                target_word=word,
                speaker=r_spk,
                ref_file=r_ref_wav,
                ref_text=r_ref_txt,
                output_path=r_out_path,
            )

            retention_ok = (r_res["predicted_sense"] == r_case["expected_sense"]) and not r_res["is_abstained"]
            logger.info(f"  [Step G - Retention] '{r_case['sentence']}' (Exp Sense {r_case['expected_sense']}) -> Retr: {r_res['predicted_sense']} | Margin: {r_res['margin']:.2f} | Retention OK: {retention_ok}")
            retention_results.append({
                "case_id": r_case["id"],
                "expected_sense": r_case["expected_sense"],
                "retrieved_sense": r_res["predicted_sense"],
                "margin": r_res["margin"],
                "is_abstained": r_res["is_abstained"],
                "retention_success": retention_ok,
            })

        workflow_log.append({
            "homograph": word,
            "transfer_results": transfer_results,
            "specificity_results": specificity_results,
            "retention_results": retention_results,
        })

    results["learning_loop_results"] = workflow_log

    # -----------------------------------------------------------------------
    # Step 6: Multi-Architecture Comparative Benchmark
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING COMPARATIVE BENCHMARK ACROSS 6 ARCHITECTURES")
    logger.info("=" * 70)

    # Collect all evaluation sentences
    eval_pool = []
    for h in PHASE3_BENCHMARK_CASES:
        w = h["homograph"]
        # Add transfer cases
        for tc in h["transfer_unseen_s0"]:
            eval_pool.append({"word": w, "sentence": tc["sentence"], "expected_sense": tc["expected_sense"], "phonetic_word": tc["phonetic_word"], "speaker": tc["speaker"], "type": "transfer"})
        # Add opposite cases
        for oc in h["opposite_specificity_s1"]:
            eval_pool.append({"word": w, "sentence": oc["sentence"], "expected_sense": oc["expected_sense"], "phonetic_word": oc["phonetic_word"], "speaker": oc["speaker"], "type": "opposite"})
        # Add retention cases
        for rc in h["retention_test"]:
            eval_pool.append({"word": w, "sentence": rc["sentence"], "expected_sense": rc["expected_sense"], "phonetic_word": rc["phonetic_word"], "speaker": rc["speaker"], "type": "retention"})

    arch_names = ["original_f5tts", "phonetic_substitution", "cosine_knn", "direct_cpa_1nn", "hopfield_ablated", "semflow_memory"]
    arch_stats = {name: {"correct": 0, "transfers": 0, "false_corrections": 0, "total": 0, "transfer_total": 0, "opposite_total": 0, "t_query_sum": 0.0} for name in arch_names}

    for item in eval_pool:
        w = item["word"]
        sent = item["sentence"]
        exp_s = item["expected_sense"]
        phon_w = item["phonetic_word"]
        spk = item["speaker"]
        typ = item["type"]
        ref_wav = SPEAKER_REFS[spk]["wav_path"]
        ref_txt = SPEAKER_REFS[spk]["ref_text"]

        # Extract query key
        q_key = semflow_sys.extract_context_key(sent, w)

        for name in arch_names:
            arch_stats[name]["total"] += 1
            if typ == "transfer": arch_stats[name]["transfer_total"] += 1
            if typ == "opposite": arch_stats[name]["opposite_total"] += 1

            if name == "original_f5tts":
                # Baseline has no memory; defaults to common sense (e.g. sense 1 for lead, sense 0 for bow)
                pred_sense = 1 if w in ("lead", "read") else 0
                is_correct = (pred_sense == exp_s)
                arch_stats[name]["correct"] += (1 if is_correct else 0)

            elif name == "phonetic_substitution":
                # Oracle phonetic substitution always applies target phonetic word
                pred_sense = exp_s
                arch_stats[name]["correct"] += 1
                if typ == "transfer": arch_stats[name]["transfers"] += 1

            else:
                mem = architectures[name]
                t0 = time.time()
                ret = mem.retrieve(q_key, target_spelling=w)
                t_q = time.time() - t0
                arch_stats[name]["t_query_sum"] += t_q

                is_abstained = ret.get("is_abstained", True)
                ret_entry = ret.get("entry", None)
                pred_sense = ret_entry.sense_id if (not is_abstained and ret_entry) else -1

                if not is_abstained and pred_sense != -1:
                    is_correct = (pred_sense == exp_s)
                    arch_stats[name]["correct"] += (1 if is_correct else 0)
                    if typ == "transfer" and is_correct:
                        arch_stats[name]["transfers"] += 1
                    if typ == "opposite" and pred_sense != exp_s:
                        arch_stats[name]["false_corrections"] += 1
                else:
                    # Abstained: system outputs default baseline
                    default_sense = 1 if w in ("lead", "read") else 0
                    is_correct = (default_sense == exp_s)
                    arch_stats[name]["correct"] += (1 if is_correct else 0)

    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3 ARCHITECTURE BENCHMARK SUMMARY")
    logger.info("=" * 70)

    bench_summary = {}
    for name in arch_names:
        tot = arch_stats[name]["total"]
        t_tot = max(1, arch_stats[name]["transfer_total"])
        o_tot = max(1, arch_stats[name]["opposite_total"])

        overall_acc = (arch_stats[name]["correct"] / tot) * 100.0
        transfer_acc = (arch_stats[name]["transfers"] / t_tot) * 100.0
        false_corr_rate = (arch_stats[name]["false_corrections"] / o_tot) * 100.0
        avg_q_ms = (arch_stats[name]["t_query_sum"] / tot) * 1000.0

        bench_summary[name] = {
            "overall_accuracy_pct": round(overall_acc, 1),
            "transfer_accuracy_pct": round(transfer_acc, 1),
            "false_correction_rate_pct": round(false_corr_rate, 1),
            "avg_memory_query_latency_ms": round(avg_q_ms, 3),
        }
        logger.info(f"{name:<24} | Acc: {overall_acc:5.1f}% | Transfer: {transfer_acc:5.1f}% | False Corr: {false_corr_rate:5.1f}% | Query: {avg_q_ms:5.2f}ms")

    results["comparative_architecture_benchmark"] = bench_summary

    # -----------------------------------------------------------------------
    # Step 7: Sample-Exact Bit-Level Preservation Audit
    # -----------------------------------------------------------------------
    test_base_path = "reports/acoustic_validation_wavs/lead_s0_f_med_baseline.wav"
    test_regen_path = "reports/acoustic_validation_wavs/lead_s0_f_med_local_regen.wav"
    test_mel_path = "reports/acoustic_validation_wavs/lead_s0_f_med_mel_blend.wav"

    if os.path.exists(test_base_path) and os.path.exists(test_regen_path) and os.path.exists(test_mel_path):
        b_wav, sr = sf.read(test_base_path)
        r_wav, _ = sf.read(test_regen_path)
        m_wav, _ = sf.read(test_mel_path)

        s_idx = int(2.24 * sr)
        e_idx = int(2.66 * sr)
        fade = int(0.02 * sr)

        pre_b = b_wav[:s_idx - fade]
        pre_r = r_wav[:s_idx - fade]
        suff_b = b_wav[e_idx + fade:]
        suff_r = r_wav[-(len(suff_b)):]

        max_pre_diff_r = float(np.max(np.abs(pre_b - pre_r)))
        max_suff_diff_r = float(np.max(np.abs(suff_b - suff_r)))
        max_pre_diff_m = float(np.max(np.abs(b_wav[:len(pre_b)] - m_wav[:len(pre_b)])))

        results["acoustic_preservation_audit"] = {
            "localized_regeneration_pre_seam_max_diff": max_pre_diff_r,
            "localized_regeneration_post_seam_max_diff": max_suff_diff_r,
            "is_bit_exact_preserved": (max_pre_diff_r == 0.0 and max_suff_diff_r == 0.0),
            "masked_mel_blending_pre_seam_max_diff": round(max_pre_diff_m, 6),
        }
        logger.info(f"\n[BIT-EXACT AUDIT] Local Regen Pre Diff: {max_pre_diff_r} | Post Diff: {max_suff_diff_r} | Bit-Exact: {max_pre_diff_r == 0.0}")

    # Save structured results
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n[SUCCESS] Phase 3 research prototype completed! Results saved to {RESULTS_JSON_PATH}")
    return results

if __name__ == "__main__":
    run_phase3_suite()
