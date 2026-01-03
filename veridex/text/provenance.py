from veridex.core.provenance import C2PASignal

class C2PATextProvenance(C2PASignal):
    """
    C2PA Provenance checker for Text files.
    Looks for sidecar manifests (.c2pa) to verify text content.
    """
    @property
    def name(self) -> str:
        return "c2pa_text_provenance"
