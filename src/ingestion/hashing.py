import hashlib
from pathlib import Path


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    """Calculate a file SHA-256 digest without loading the whole file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_document_id(sha256: str) -> str:
    """Return a stable document identifier based on its content hash."""
    return f"ma_doc_{sha256[:24]}"


def stable_chunk_id(document_id: str, chunk_index: int, text: str) -> str:
    """Return a stable chunk identifier."""
    payload = f"{document_id}:{chunk_index}:{text}".encode()
    return hashlib.sha256(payload).hexdigest()[:32]
