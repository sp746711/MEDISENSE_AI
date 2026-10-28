from pathlib import Path
import onnxruntime as ort

path = Path("models/xray/bodypart_full/model.onnx")
session = ort.InferenceSession(
    str(path),
    providers=["CPUExecutionProvider"]
)
meta = session.get_modelmeta()

print("MODEL:", path.resolve())
print("PROVIDERS:", session.get_providers())
print("INPUTS:", [(x.name, x.shape, x.type) for x in session.get_inputs()])
print("OUTPUTS:", [(x.name, x.shape, x.type) for x in session.get_outputs()])
print("CUSTOM METADATA:", meta.custom_metadata_map)
print("GRAPH NAME:", meta.graph_name)
print("PRODUCER:", meta.producer_name)
print("DOMAIN:", meta.domain)
print("MODEL VERSION:", meta.version)
print("DESCRIPTION:", meta.description)
