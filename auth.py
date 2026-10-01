"""Simple authentication and role-based access control (Chapter 3, Section 3.9).
Passwords are stored as salted PBKDF2 hashes. For production, use your institution's identity provider."""
import hashlib, hmac, json, os, datetime, csv

def _secret(name, default):
    """Reads a password from Streamlit secrets or an environment variable; falls back to the demo default."""
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


DEMO = {"ADMIN_PASSWORD": "ChangeMe-Admin1", "MANAGER_PASSWORD": "ChangeMe-Manager1", "SUPERVISOR_PASSWORD": "ChangeMe-Super1"}


def using_demo_passwords():
    return all(_secret(k, v) == v for k, v in DEMO.items())


USERS_FILE = "data/users.json"
AUDIT_FILE = "data/audit_log.csv"
ROLES = {"admin": ["dashboard", "alerts", "simulator", "audit"],
         "manager": ["dashboard", "alerts", "simulator"],
         "supervisor": ["dashboard", "alerts"]}


def _hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000).hex()


def create_user(users, name, password, role):
    salt = os.urandom(16).hex()
    users[name] = {"salt": salt, "hash": _hash(password, salt), "role": role}


def load_users():
    if not os.path.exists(USERS_FILE):
        os.makedirs("data", exist_ok=True)
        users = {}
        # Set ADMIN_PASSWORD / MANAGER_PASSWORD / SUPERVISOR_PASSWORD as secrets or environment variables.
        # The defaults below are DEMO ONLY and are public if this code is on GitHub.
        create_user(users, "admin", _secret("ADMIN_PASSWORD", DEMO["ADMIN_PASSWORD"]), "admin")
        create_user(users, "manager", _secret("MANAGER_PASSWORD", DEMO["MANAGER_PASSWORD"]), "manager")
        create_user(users, "supervisor", _secret("SUPERVISOR_PASSWORD", DEMO["SUPERVISOR_PASSWORD"]), "supervisor")
        json.dump(users, open(USERS_FILE, "w"))
    return json.load(open(USERS_FILE))


def verify(username, password):
    u = load_users().get(username)
    if not u:
        return None
    return u["role"] if hmac.compare_digest(_hash(password, u["salt"]), u["hash"]) else None


def audit(user, action):
    os.makedirs("data", exist_ok=True)
    new = not os.path.exists(AUDIT_FILE)
    with open(AUDIT_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "user", "action"])
        w.writerow([datetime.datetime.now().isoformat(timespec="seconds"), user, action])
