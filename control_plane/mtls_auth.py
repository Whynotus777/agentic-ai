# control_plane/mtls_auth.py
"""
Mutual TLS (mTLS) implementation for service-to-service authentication.
Provides zero-trust networking between all internal services.
"""

import asyncio
import ssl
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from cryptography import x509
from cryptography.x509.oid import NameOID, ExtensionOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
import aiohttp
from aiohttp import web

from opentelemetry import trace
tracer = trace.get_tracer(__name__)


@dataclass
class ServiceIdentity:
    """Service identity for mTLS"""
    service_name: str
    service_id: str
    namespace: str
    common_name: str
    certificate: x509.Certificate
    private_key: rsa.RSAPrivateKey
    issued_at: datetime
    expires_at: datetime
    fingerprint: str
    allowed_endpoints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TLSPolicy:
    """TLS policy for service communication"""
    min_tls_version: str = "TLSv1.3"
    cipher_suites: List[str] = field(default_factory=list)
    require_client_cert: bool = True
    verify_hostname: bool = True
    cert_rotation_days: int = 90
    allowed_sans: List[str] = field(default_factory=list)  # Subject Alternative Names


class MTLSManager:
    """
    Manages mTLS certificates and authentication for service-to-service communication
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ca_cert = None
        self.ca_key = None
        self.service_identities: Dict[str, ServiceIdentity] = {}
        self.trust_store: Dict[str, x509.Certificate] = {}
        self.revoked_certs: Set[str] = set()
        self.policies: Dict[str, TLSPolicy] = {}
        self._init_ca()
        self._init_policies()
        
    def _init_ca(self):
        """Initialize Certificate Authority"""
        # Generate CA key
        self.ca_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
            backend=default_backend()
        )
        
        # Generate CA certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "California"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Agentic AI"),
            x509.NameAttribute(NameOID.COMMON_NAME, "Agentic AI Internal CA")
        ])
        
        self.ca_cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(self.ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=3650))  # 10 years
            .add_extension(
                x509.BasicConstraints(ca=True, path_length=None),
                critical=True
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=True,
                    crl_sign=True,
                    encipher_only=False,
                    decipher_only=False
                ),
                critical=True
            )
            .sign(self.ca_key, hashes.SHA256(), backend=default_backend())
        )
    
    def _init_policies(self):
        """Initialize TLS policies"""
        # Default policy
        self.policies["default"] = TLSPolicy(
            min_tls_version="TLSv1.3",
            cipher_suites=[
                "TLS_AES_256_GCM_SHA384",
                "TLS_AES_128_GCM_SHA256",
                "TLS_CHACHA20_POLY1305_SHA256"
            ],
            require_client_cert=True,
            verify_hostname=True,
            cert_rotation_days=90
        )
        
        # High security policy for sensitive services
        self.policies["high_security"] = TLSPolicy(
            min_tls_version="TLSv1.3",
            cipher_suites=["TLS_AES_256_GCM_SHA384"],
            require_client_cert=True,
            verify_hostname=True,
            cert_rotation_days=30
        )
    
    @tracer.start_as_current_span("issue_service_certificate")
    async def issue_service_certificate(
        self,
        service_name: str,
        namespace: str = "default",
        allowed_endpoints: List[str] = None
    ) -> ServiceIdentity:
        """
        Issue a certificate for a service
        """
        span = trace.get_current_span()
        
        # Generate service key
        service_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        # Create certificate
        service_id = str(uuid.uuid4())
        common_name = f"{service_name}.{namespace}.svc.cluster.local"
        
        subject = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Agentic AI"),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, namespace),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name)
        ])
        
        # Build certificate
        builder = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(self.ca_cert.issuer)
            .public_key(service_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=90))
        )
        
        # Add Subject Alternative Names
        sans = [
            x509.DNSName(common_name),
            x509.DNSName(service_name),
            x509.DNSName(f"{service_name}.{namespace}"),
            x509.DNSName(f"*.{namespace}.svc.cluster.local")
        ]
        
        builder = builder.add_extension(
            x509.SubjectAlternativeName(sans),
            critical=False
        )
        
        # Add key usage
        builder = builder.add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False
            ),
            critical=True
        )
        
        # Add extended key usage
        builder = builder.add_extension(
            x509.ExtendedKeyUsage([
                x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
                x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH
            ]),
            critical=True
        )
        
        # Add custom extension for service metadata
        service_metadata = {
            "service_id": service_id,
            "service_name": service_name,
            "namespace": namespace,
            "issued_at": datetime.utcnow().isoformat()
        }
        
        # Sign certificate
        certificate = builder.sign(self.ca_key, hashes.SHA256(), backend=default_backend())
        
        # Calculate fingerprint
        fingerprint = hashlib.sha256(
            certificate.public_bytes(serialization.Encoding.DER)
        ).hexdigest()
        
        # Create service identity
        identity = ServiceIdentity(
            service_name=service_name,
            service_id=service_id,
            namespace=namespace,
            common_name=common_name,
            certificate=certificate,
            private_key=service_key,
            issued_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=90),
            fingerprint=fingerprint,
            allowed_endpoints=allowed_endpoints or [],
            metadata=service_metadata
        )
        
        # Store identity
        self.service_identities[service_id] = identity
        self.trust_store[fingerprint] = certificate
        
        span.set_attributes({
            "mtls.service_name": service_name,
            "mtls.service_id": service_id,
            "mtls.fingerprint": fingerprint,
            "mtls.expires_at": identity.expires_at.isoformat()
        })
        
        return identity
    
    async def validate_certificate(
        self,
        cert_pem: bytes,
        expected_service: Optional[str] = None
    ) -> Tuple[bool, Optional[ServiceIdentity]]:
        """
        Validate a certificate presented by a service
        
        Returns:
            Tuple of (is_valid, service_identity)
        """
        try:
            # Parse certificate
            cert = x509.load_pem_x509_certificate(cert_pem, backend=default_backend())
            
            # Calculate fingerprint
            fingerprint = hashlib.sha256(
                cert.public_bytes(serialization.Encoding.DER)
            ).hexdigest()
            
            # Check if revoked
            if fingerprint in self.revoked_certs:
                return False, None
            
            # Check if in trust store
            if fingerprint not in self.trust_store:
                return False, None
            
            # Verify certificate chain
            if not self._verify_certificate_chain(cert):
                return False, None
            
            # Check expiration
            if datetime.utcnow() > cert.not_valid_after:
                return False, None
            
            # Extract service name from CN
            cn = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
            service_name = cn.split('.')[0]
            
            # Check expected service if provided
            if expected_service and service_name != expected_service:
                return False, None
            
            # Find service identity
            for sid, identity in self.service_identities.items():
                if identity.fingerprint == fingerprint:
                    return True, identity
            
            # Certificate is valid but identity not found locally
            # (might be from another instance)
            return True, None
            
        except Exception as e:
            print(f"Certificate validation error: {e}")
            return False, None
    
    def _verify_certificate_chain(self, cert: x509.Certificate) -> bool:
        """Verify certificate is signed by our CA"""
        try:
            # In production, use proper chain validation
            # For now, check if issuer matches our CA
            return cert.issuer == self.ca_cert.subject
        except Exception:
            return False
    
    async def create_mtls_context(
        self,
        service_identity: ServiceIdentity,
        policy_name: str = "default"
    ) -> ssl.SSLContext:
        """
        Create SSL context for mTLS communication
        """
        policy = self.policies.get(policy_name, self.policies["default"])
        
        # Create SSL context
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        
        # Set TLS version
        if policy.min_tls_version == "TLSv1.3":
            context.minimum_version = ssl.TLSVersion.TLSv1_3
        
        # Set cipher suites
        if policy.cipher_suites:
            context.set_ciphers(':'.join(policy.cipher_suites))
        
        # Load certificate and key
        cert_pem = service_identity.certificate.public_bytes(serialization.Encoding.PEM)
        key_pem = service_identity.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        # Create temporary files (in production, use memory)
        import tempfile
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as cert_file:
            cert_file.write(cert_pem)
            cert_path = cert_file.name
        
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as key_file:
            key_file.write(key_pem)
            key_path = key_file.name
        
        context.load_cert_chain(cert_path, key_path)
        
        # Load CA certificate
        ca_pem = self.ca_cert.public_bytes(serialization.Encoding.PEM)
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as ca_file:
            ca_file.write(ca_pem)
            ca_path = ca_file.name
        
        context.load_verify_locations(ca_path)
        
        # Set verification
        if policy.require_client_cert:
            context.verify_mode = ssl.CERT_REQUIRED
        else:
            context.verify_mode = ssl.CERT_OPTIONAL
        
        if policy.verify_hostname:
            context.check_hostname = True
        
        return context
    
    async def make_secure_request(
        self,
        service_identity: ServiceIdentity,
        target_service: str,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make an mTLS-secured request to another service
        """
        # Create SSL context
        ssl_context = await self.create_mtls_context(service_identity)
        
        # Create connector with SSL context
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        
        # Build URL
        url = f"https://{target_service}.{service_identity.namespace}.svc.cluster.local{endpoint}"
        
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.request(
                method,
                url,
                json=data,
                headers={
                    "X-Service-Name": service_identity.service_name,
                    "X-Service-ID": service_identity.service_id,
                    "X-Request-ID": str(uuid.uuid4())
                }
            ) as response:
                return await response.json()
    
    async def rotate_certificate(
        self,
        service_id: str
    ) -> ServiceIdentity:
        """
        Rotate certificate for a service
        """
        if service_id not in self.service_identities:
            raise ValueError(f"Service {service_id} not found")
        
        old_identity = self.service_identities[service_id]
        
        # Issue new certificate
        new_identity = await self.issue_service_certificate(
            service_name=old_identity.service_name,
            namespace=old_identity.namespace,
            allowed_endpoints=old_identity.allowed_endpoints
        )
        
        # Revoke old certificate
        self.revoked_certs.add(old_identity.fingerprint)
        
        # Remove from trust store
        del self.trust_store[old_identity.fingerprint]
        
        print(f"Rotated certificate for {old_identity.service_name}")
        
        return new_identity
    
    async def check_certificate_expiry(self):
        """
        Check for certificates nearing expiry and rotate them
        """
        while True:
            try:
                now = datetime.utcnow()
                
                for service_id, identity in list(self.service_identities.items()):
                    # Check if expires in next 7 days
                    days_until_expiry = (identity.expires_at - now).days
                    
                    if days_until_expiry <= 7:
                        print(f"Certificate for {identity.service_name} expires in {days_until_expiry} days")
                        
                        # Rotate certificate
                        await self.rotate_certificate(service_id)
                
            except Exception as e:
                print(f"Certificate expiry check error: {e}")
            
            await asyncio.sleep(86400)  # Check daily
    
    def export_ca_certificate(self) -> bytes:
        """Export CA certificate for distribution"""
        return self.ca_cert.public_bytes(serialization.Encoding.PEM)
    
    def export_service_certificate(self, service_id: str) -> Tuple[bytes, bytes]:
        """
        Export service certificate and key
        
        Returns:
            Tuple of (certificate_pem, key_pem)
        """
        if service_id not in self.service_identities:
            raise ValueError(f"Service {service_id} not found")
        
        identity = self.service_identities[service_id]
        
        cert_pem = identity.certificate.public_bytes(serialization.Encoding.PEM)
        key_pem = identity.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        return cert_pem, key_pem


class MTLSMiddleware:
    """
    Middleware for enforcing mTLS on incoming requests
    """
    
    def __init__(self, mtls_manager: MTLSManager):
        self.mtls_manager = mtls_manager
    
    @web.middleware
    async def middleware(self, request: web.Request, handler):
        """Validate mTLS certificate for incoming requests"""
        
        # Extract certificate from request
        peer_cert = request.transport.get_extra_info('peercert')
        
        if not peer_cert:
            return web.json_response(
                {"error": "Client certificate required"},
                status=401
            )
        
        # Convert to PEM format
        import ssl
        cert_pem = ssl.DER_cert_to_PEM_cert(peer_cert)
        
        # Validate certificate
        is_valid, identity = await self.mtls_manager.validate_certificate(
            cert_pem.encode()
        )
        
        if not is_valid:
            return web.json_response(
                {"error": "Invalid client certificate"},
                status=403
            )
        
        # Add service identity to request
        request["service_identity"] = identity
        
        # Process request
        return await handler(request)


# Service mesh integration
class ServiceMesh:
    """
    Service mesh for managing service-to-service communication
    """
    
    def __init__(self, mtls_manager: MTLSManager):
        self.mtls_manager = mtls_manager
        self.service_registry: Dict[str, Dict[str, Any]] = {}
        self.load_balancers: Dict[str, Any] = {}
        
    async def register_service(
        self,
        service_name: str,
        endpoints: List[str],
        health_check: str = "/health"
    ):
        """Register a service in the mesh"""
        
        # Issue certificate
        identity = await self.mtls_manager.issue_service_certificate(
            service_name=service_name,
            allowed_endpoints=endpoints
        )
        
        # Register service
        self.service_registry[service_name] = {
            "identity": identity,
            "endpoints": endpoints,
            "health_check": health_check,
            "instances": [],
            "circuit_breaker": {
                "failure_threshold": 5,
                "timeout": 60,
                "state": "closed"
            }
        }
        
        print(f"Registered service {service_name} in mesh")
        
        return identity
    
    async def discover_service(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Discover a service in the mesh"""
        return self.service_registry.get(service_name)
    
    async def call_service(
        self,
        caller_identity: ServiceIdentity,
        target_service: str,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Call another service through the mesh with circuit breaking
        and load balancing
        """
        service_info = await self.discover_service(target_service)
        
        if not service_info:
            raise ValueError(f"Service {target_service} not found")
        
        # Check circuit breaker
        if service_info["circuit_breaker"]["state"] == "open":
            raise Exception(f"Circuit breaker open for {target_service}")
        
        try:
            # Make mTLS request
            result = await asyncio.wait_for(
                self.mtls_manager.make_secure_request(
                    caller_identity,
                    target_service,
                    endpoint,
                    method,
                    data
                ),
                timeout=timeout
            )
            
            # Reset circuit breaker on success
            service_info["circuit_breaker"]["failure_count"] = 0
            
            return result
            
        except Exception as e:
            # Update circuit breaker
            cb = service_info["circuit_breaker"]
            cb["failure_count"] = cb.get("failure_count", 0) + 1
            
            if cb["failure_count"] >= cb["failure_threshold"]:
                cb["state"] = "open"
                cb["opened_at"] = datetime.utcnow()
                
                # Schedule circuit breaker reset
                asyncio.create_task(
                    self._reset_circuit_breaker(target_service, cb["timeout"])
                )
            
            raise
    
    async def _reset_circuit_breaker(self, service_name: str, timeout: int):
        """Reset circuit breaker after timeout"""
        await asyncio.sleep(timeout)
        
        if service_name in self.service_registry:
            self.service_registry[service_name]["circuit_breaker"]["state"] = "half-open"
            print(f"Circuit breaker for {service_name} moved to half-open")


# Example usage
async def main():
    config = {}
    
    # Initialize mTLS manager
    mtls_manager = MTLSManager(config)
    
    # Issue certificates for services
    api_gateway = await mtls_manager.issue_service_certificate("api-gateway")
    orchestrator = await mtls_manager.issue_service_certificate("orchestrator")
    execution_layer = await mtls_manager.issue_service_certificate("execution-layer")
    
    print(f"API Gateway cert: {api_gateway.fingerprint}")
    print(f"Orchestrator cert: {orchestrator.fingerprint}")
    print(f"Execution Layer cert: {execution_layer.fingerprint}")
    
    # Initialize service mesh
    mesh = ServiceMesh(mtls_manager)
    
    # Register services
    await mesh.register_service(
        "api-gateway",
        ["/api/v1/*"],
        "/health"
    )
    
    await mesh.register_service(
        "orchestrator",
        ["/orchestrate", "/status"],
        "/health"
    )
    
    # Simulate service call
    try:
        result = await mesh.call_service(
            api_gateway,
            "orchestrator",
            "/orchestrate",
            "POST",
            {"task": "example"}
        )
        print(f"Service call result: {result}")
    except Exception as e:
        print(f"Service call failed: {e}")
    
    # Export CA cert for distribution
    ca_cert = mtls_manager.export_ca_certificate()
    print(f"CA Certificate:\n{ca_cert.decode()[:200]}...")
    
    # Start certificate rotation monitor
    asyncio.create_task(mtls_manager.check_certificate_expiry())


if __name__ == "__main__":
    asyncio.run(main())