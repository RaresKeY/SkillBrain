import uuid, random, json, gzip, os, hashlib, re, base64, hmac
from datetime import datetime, timezone

# --- config ---
FMT = "jsonl"            # "jsonl" or "json"
COMPRESS = False         # True => gzip, False => plain
COUNT = 200
BASENAME = f"friends_db_{COUNT}"  # file name without extension

# --- helpers ---
def date_to_unix(day, month, year):
    dt = datetime(int(year), int(month), int(day), tzinfo=timezone.utc)
    return int(dt.timestamp())

# phone number helper functions
def get_secret_key() -> bytes:
    key_b64 = os.environ.get("PHONE_HMAC_KEY")
    if not key_b64:
        raise RuntimeError("PHONE_HMAC_KEY not set in environment")
    return base64.urlsafe_b64decode(key_b64.encode())

SECRET_KEY = get_secret_key()

# phone number helper functions, for user input, not for stored hash
def is_valid_e164(number: str) -> bool:
    # E.164: phone numbers must start with +, followed by 8–15 digits (country code + subscriber), no spaces/dashes
    return bool(re.fullmatch(r"\+[1-9]\d{7,14}", number))

# one way encryption
def hmac_phone(number:str) -> str:
    return hmac.new(SECRET_KEY, number.encode(), hashlib.sha256).hexdigest()

# verify plaintext vs hashed
def verify_phone(number: str, hashed: str) -> bool:
    new_hash = hmac.new(SECRET_KEY, number.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(new_hash, hashed)

# HOBBIES_POOL = [
#     "golf","gaming","flying","chess","reading","running","coding",
#     "kpop","music","cycling","hiking","swimming","movies","art"
# ]

import json

with open("/home/mintmainog/workspace/VS Code Workspaces/SkillBrain_Python_Homework_Fork/sessions/homework/hw_adv/tema_4/data/hobbies.json", "r", encoding="utf-8") as f:
    HOBBIES_POOL = json.load(f)

NAMES_FIRST = ["Paul","Bob","Alice","Eve","Carol","Dave","Tasman","John","Mary","Tom","Linda"]
NAMES_LAST  = ["Schindler","Smith","Johnson","Williams","Brown","Jones","Miller","Davis","Wilson","Taylor","Liberal"]

def generate_random_friend():
    name = f"{random.choice(NAMES_FIRST)} {random.choice(NAMES_LAST)}"
    phone_number = f"55501{random.randint(0,99):02d}" # 555-0100..0199 (repeats allowed)
    e164_number = f"+1{phone_number}" # chose US region
    if not is_valid_e164(e164_number):
        raise ValueError(f"Invalid E.164 phone number: {e164_number}")
    # DOB 1950..2010 (safe day 1..28)
    dob = date_to_unix(random.randint(1,28), random.randint(1,12), random.randint(1950,2010))
    # Last contact 2011..2024
    last_contact = date_to_unix(random.randint(1,28), random.randint(1,12), random.randint(2011,2024))
    hobbies = random.sample(HOBBIES_POOL, k=random.randint(1,4))
    distance = round(random.uniform(0.0, 50.0), 2)
    return {
        "name": name,
        "phone_number": {"hmac": hmac_phone(e164_number)}, # TODO: allow 'None' or empty for Unknown
        "dob": dob, # TODO: allow 'None' for Unknown
        "last_contact": last_contact, # allow 'None' to sim site-wide db (record site interactions), update main code to handle 'None'
        "hobbies": hobbies,
        "distance": distance, # TODO: allow 'None' for Unknown
        "connection": 0.0 # TODO: expand connections
    }

def _open(path, text=True):
    if COMPRESS:
        return gzip.open(path, "wt" if text else "wb", encoding="utf-8")
    return open(path, "w" if text else "wb", encoding="utf-8" if text else None)

def generate_users_stream():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ext = ".jsonl" if FMT == "jsonl" else ".json"
    path = os.path.join(base_dir, BASENAME + ext + (".gz" if COMPRESS else ""))

    if FMT == "jsonl":
        with _open(path, text=True) as f:
            for _ in range(COUNT):
                uid = str(uuid.uuid4())
                rec = {uid: generate_random_friend()}
                f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
    else:  # FMT == "json" -> big dict streamed without holding all in RAM
        with _open(path, text=True) as f:
            f.write("{")
            first = True
            for _ in range(COUNT):
                uid = str(uuid.uuid4())
                if not first:
                    f.write(",")
                else:
                    first = False
                f.write(json.dumps(uid))  # writes the UUID key with quotes
                f.write(":")
                f.write(json.dumps(generate_random_friend(), ensure_ascii=False, separators=(",", ":")))
            f.write("}")

    print(f"Saved {COUNT:,} users to {path}")

if __name__ == "__main__":
    generate_users_stream()
