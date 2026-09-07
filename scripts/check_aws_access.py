"""
AWS & S3 Diagnostic Utility for FlowEdit

Checks:
1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3 bucket names)
2. AWS STS caller identity (Account ID, User/Role ARN)
3. S3 bucket reachability and PutObject/GetObject permissions for FlowEdit dictionary
"""

import os
import sys


def mask_key(key: str) -> str:
    if not key:
        return "<not set>"
    if len(key) <= 8:
        return key[:2] + "****"
    return key[:4] + "****" + key[-4:]


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from flowedit.utils.env import load_flowedit_env
    env_loaded = load_flowedit_env()
except Exception:
    env_loaded = False


def check_aws_credentials():
    print("=" * 60)
    print(" FlowEdit AWS & S3 Access Diagnostic")
    print("=" * 60)
    print(f"Loaded .env file: {'YES' if env_loaded else 'NO / Not found'}")

    # 1. Inspect Environment Variables
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    session_token = os.environ.get("AWS_SESSION_TOKEN")
    region = (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-east-1"
    )
    bucket = (
        os.environ.get("FLOWEDIT_S3_BUCKET")
        or os.environ.get("S3_BUCKET_NAME")
        or os.environ.get("AWS_S3_BUCKET")
        or "flowedit-bucket"
    )
    prefix = os.environ.get("FLOWEDIT_S3_PREFIX", "corrections/")

    print("\n[1] Environment Variables:")
    print(f"  - AWS_ACCESS_KEY_ID     : {mask_key(access_key)} (length: {len(access_key) if access_key else 0})")
    print(f"  - AWS_SECRET_ACCESS_KEY : {'[SET]' if secret_key else '<not set>'}")
    print(f"  - AWS_SESSION_TOKEN     : {'[SET]' if session_token else '<not set>'}")
    print(f"  - AWS_REGION            : {region}")
    print(f"  - Target S3 Bucket      : {bucket}")
    print(f"  - S3 Key Prefix         : {prefix}")

    # Check ~/.aws files
    home_dir = os.path.expanduser("~")
    aws_dir = os.path.join(home_dir, ".aws")
    cred_file = os.path.join(aws_dir, "credentials")
    conf_file = os.path.join(aws_dir, "config")
    print("\n[2] AWS Config Files on Disk:")
    print(f"  - {cred_file} : {'EXISTS' if os.path.exists(cred_file) else 'NOT FOUND'}")
    print(f"  - {conf_file} : {'EXISTS' if os.path.exists(conf_file) else 'NOT FOUND'}")

    active_profile = os.environ.get("AWS_PROFILE", "default")
    print(f"  - Active Profile in Use : {active_profile}")

    # Read profiles from ~/.aws/credentials if present
    import configparser
    if os.path.exists(cred_file):
        cp = configparser.ConfigParser()
        try:
            cp.read(cred_file)
            print(f"  - Profiles found in credentials file: {list(cp.sections())}")
            for sec in cp.sections():
                ak = cp.get(sec, "aws_access_key_id", fallback="<none>")
                tok = cp.get(sec, "aws_session_token", fallback=None)
                print(f"    * [{sec}] key: {mask_key(ak)} (type: {'temporary (ASIA)' if ak.startswith('ASIA') else 'permanent (AKIA)' if ak.startswith('AKIA') else 'unknown'}) | session_token: {'SET' if tok else 'NOT SET'}")
        except Exception as e:
            print(f"    (Could not parse credentials file: {e})")

    # 2. Check boto3 and AWS STS Caller Identity
    print("\n[3] Verifying AWS Identity via STS (boto3)...")
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
    except ImportError:
        print("  [FAIL] Error: 'boto3' is not installed in this Python environment.")
        print("    Install it via: pip install boto3")
        return

    # Check active session identity
    identity = None
    try:
        session = boto3.Session(region_name=region)
        sts_client = session.client("sts")
        identity = sts_client.get_caller_identity()
        print("  [OK] STS Authentication SUCCESSFUL for active profile!")
        print(f"    - AWS Account ID : {identity.get('Account')}")
        print(f"    - IAM Identity   : {identity.get('Arn')}")
        print(f"    - User ID        : {identity.get('UserId')}")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        msg = e.response.get("Error", {}).get("Message")
        print(f"  [FAIL] STS Authentication FAILED for active profile ({active_profile}): [{code}] {msg}")
        if code in ("InvalidClientTokenId", "InvalidAccessKeyId"):
            print("\n  >>> ROOT CAUSE: The credentials loaded from /home/rsurya/.aws/credentials")
            print("      are INVALID or EXPIRED.")
            print("      - If key starts with 'ASIA': Temporary session token expired or missing.")
            print("      - If key starts with 'AKIA': IAM Access Key was deleted or deactivated in AWS.")
            print("      - Or the key belongs to an AWS account that was closed/replaced.")
    except NoCredentialsError:
        print("  [FAIL] No AWS credentials found via env, ~/.aws/credentials, or IAM role.")
    except Exception as e:
        print(f"  [FAIL] Unexpected STS error: {e}")

    # If active profile failed, check other available profiles
    try:
        session = boto3.Session()
        all_profiles = session.available_profiles
        if len(all_profiles) > 1 or (all_profiles and active_profile not in all_profiles):
            print("\n  Checking other available AWS profiles:")
            for p in all_profiles:
                if p == active_profile:
                    continue
                try:
                    p_session = boto3.Session(profile_name=p, region_name=region)
                    p_sts = p_session.client("sts")
                    p_id = p_sts.get_caller_identity()
                    print(f"    [OK] Profile '{p}': Account {p_id.get('Account')} ({p_id.get('Arn')})")
                except Exception:
                    print(f"    [FAIL] Profile '{p}': Invalid credentials")
    except Exception:
        pass

    if identity is None:
        print("\n>>> Cannot proceed to S3 check without valid AWS credentials. <<<")
        print("Run 'cat ~/.aws/credentials' to see the current keys, or update them.")
        return

    # 3. Check S3 Bucket Access
    print(f"\n[4] Testing Access to S3 Bucket '{bucket}'...")
    try:
        s3_client = boto3.client("s3", region_name=region)
        s3_client.head_bucket(Bucket=bucket)
        print(f"  [OK] S3 Bucket '{bucket}' exists and is accessible!")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchBucket"):
            print(f"  [FAIL] Bucket '{bucket}' does not exist in account {identity.get('Account')}.")
        elif code in ("403", "AccessDenied"):
            print(f"  [FAIL] Access Denied: Identity {identity.get('Arn')} does NOT have permission to access '{bucket}'.")
        else:
            print(f"  [FAIL] S3 head_bucket error: {e}")
        return
    except Exception as e:
        print(f"  [FAIL] S3 check error: {e}")
        return

    # 4. Test Write Permission (PutObject)
    test_key = f"{prefix.rstrip('/')}/.connectivity_test.json"
    print(f"\n[5] Testing Write Permission (PutObject) to 's3://{bucket}/{test_key}'...")
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=test_key,
            Body=b'{"status": "ok", "flowedit_test": true}',
            ContentType="application/json",
        )
        print(f"  [OK] Successfully wrote test object to s3://{bucket}/{test_key}!")

        # Test GetObject
        resp = s3_client.get_object(Bucket=bucket, Key=test_key)
        _ = resp["Body"].read()
        print(f"  [OK] Successfully verified read access (GetObject)!")

        # Clean up test object
        try:
            s3_client.delete_object(Bucket=bucket, Key=test_key)
            print("  [OK] Successfully cleaned up test object.")
        except ClientError as de:
            print(f"  [INFO] Cleanup note: User lacks 's3:DeleteObject' permission.")
            print("         This is normal; FlowEdit only requires 's3:PutObject' and 's3:GetObject'.")

        print("\n>>> S3 ACCESS VERIFICATION COMPLETE: ALL CHECKS PASSED! <<<")
    except ClientError as e:
        print(f"  [FAIL] Write test failed: {e}")


if __name__ == "__main__":
    check_aws_credentials()
