"""Near-duplicate text detection with MinHash LSH (datasketch) and a brute-force Jaccard fallback."""

from __future__ import annotations

from video_dataset.utils.text import jaccard, word_shingles


class NearDuplicateIndex:
    def __init__(self, threshold: float = 0.9, num_perm: int = 128, shingle_size: int = 3):
        self.threshold = threshold
        self.num_perm = num_perm
        self.k = shingle_size
        self._shingles: dict[str, set[str]] = {}
        self._lsh = None
        self._minhash_cls = None
        try:
            from datasketch import MinHash, MinHashLSH

            self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
            self._minhash_cls = MinHash
        except Exception:
            self._lsh = None

    def _mh(self, shingles: set[str]):  # type: ignore[no-untyped-def]
        assert self._minhash_cls is not None
        m = self._minhash_cls(num_perm=self.num_perm)
        for s in shingles:
            m.update(s.encode("utf-8"))
        return m

    def query(self, text: str) -> list[str]:
        sh = word_shingles(text, self.k)
        if not sh:
            return []
        if self._lsh is not None:
            cands = list(self._lsh.query(self._mh(sh)))
        else:
            cands = list(self._shingles.keys())
        return [c for c in cands if jaccard(sh, self._shingles[c]) >= self.threshold]

    def add(self, item_id: str, text: str) -> None:
        sh = word_shingles(text, self.k)
        self._shingles[item_id] = sh
        if self._lsh is not None and sh:
            self._lsh.insert(item_id, self._mh(sh))

    def similarity(self, a: str, b: str) -> float:
        return jaccard(self._shingles.get(a, set()), self._shingles.get(b, set()))


def near_duplicates(items: list[tuple[str, str]], threshold: float = 0.9, num_perm: int = 128, shingle_size: int = 3) -> dict[str, str]:
    """{duplicate_id: canonical_id} for texts whose shingle Jaccard >= threshold."""
    index = NearDuplicateIndex(threshold, num_perm, shingle_size)
    dups: dict[str, str] = {}
    for item_id, text in items:
        matches = index.query(text)
        if matches:
            canon = matches[0]
            dups[item_id] = dups.get(canon, canon)
        else:
            index.add(item_id, text)
    return dups
