from veridex.core.provenance import C2PASignal

class C2PAAudioProvenance(C2PASignal):
    """
    C2PA Provenance checker for Audio files.
    """
    @property
    def name(self) -> str:
        return "c2pa_audio_provenance"
