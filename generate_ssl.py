"""
Genera un certificado SSL auto-firmado para uso local/intranet.
Crea los archivos ssl/cert.pem y ssl/key.pem que Streamlit necesita.

Uso:
    python generate_ssl.py

Requiere: pip install cryptography
"""
import os
import datetime
import ipaddress

from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa

# ── Configuración ────────────────────────────────────────────────
CERT_DIR   = "ssl"
CERT_FILE  = os.path.join(CERT_DIR, "cert.pem")
KEY_FILE   = os.path.join(CERT_DIR, "key.pem")
VALID_DAYS = 3650          # 10 años
HOSTNAME   = "localhost"   # Cambia a tu IP/hostname real si es necesario
# ─────────────────────────────────────────────────────────────────

os.makedirs(CERT_DIR, exist_ok=True)

# 1. Generar clave privada RSA 2048
key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

# 2. Construir el certificado
subject = issuer = x509.Name([
    x509.NameAttribute(NameOID.COUNTRY_NAME,             "MX"),
    x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME,   "Jalisco"),
    x509.NameAttribute(NameOID.LOCALITY_NAME,            "Guadalajara"),
    x509.NameAttribute(NameOID.ORGANIZATION_NAME,        "Kimball Electronics"),
    x509.NameAttribute(NameOID.COMMON_NAME,              HOSTNAME),
])

now = datetime.datetime.utcnow()
cert = (
    x509.CertificateBuilder()
    .subject_name(subject)
    .issuer_name(issuer)
    .public_key(key.public_key())
    .serial_number(x509.random_serial_number())
    .not_valid_before(now)
    .not_valid_after(now + datetime.timedelta(days=VALID_DAYS))
    .add_extension(
        x509.SubjectAlternativeName([
            x509.DNSName("localhost"),
            x509.DNSName(HOSTNAME),
            x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        ]),
        critical=False,
    )
    .add_extension(
        x509.BasicConstraints(ca=True, path_length=None),
        critical=True,
    )
    .sign(key, hashes.SHA256())
)

# 3. Escribir clave privada (sin contraseña)
with open(KEY_FILE, "wb") as f:
    f.write(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))

# 4. Escribir certificado
with open(CERT_FILE, "wb") as f:
    f.write(cert.public_bytes(serialization.Encoding.PEM))

print(f"✅ Certificado generado:")
print(f"   Clave  : {KEY_FILE}")
print(f"   Cert   : {CERT_FILE}")
print(f"   Válido : {VALID_DAYS} días ({now.date()} → {(now + datetime.timedelta(days=VALID_DAYS)).date()})")
print()
print("Para ejecutar la app con HTTPS:")
print("   streamlit run app.py")
print()
print("⚠️  El navegador mostrará advertencia de certificado no confiable (es auto-firmado).")
print("   Haz clic en 'Avanzado → Continuar' para acceder.")
print("   Para evitar la advertencia, importa ssl/cert.pem como CA de confianza en tu sistema.")
