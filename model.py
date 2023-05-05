from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torch
from PIL import Image, ImageDraw
import requests
import pandas as pd

from hook import ProfHook

def get_image():
    image = Image.open("000000039769.jpg")
    image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-base")
    print(type(image_processor))
    inputs = image_processor(images=image, return_tensors="pt")

    return inputs

def test_yolos(model):
    model = model.to("cuda")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-base")


    inputs = image_processor(images=image, return_tensors="pt")
    inputs.to("cuda")

    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
        0
    ]

    img = ImageDraw.Draw(image)

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )

        img.rectangle(box, outline ="green", width=3)

    image.save("example.jpg")

    #print(model)



def profile_yolo(level:int = 0):
    model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-base").eval()
    hooks = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    image = get_image()
    image = image['pixel_values'].to(device)

    print(image.shape)

    hook = ProfHook()

    if level == 0:
        hooks.append(model.register_forward_pre_hook(hook.pre(f"yolos")))
        hooks.append(model.register_forward_hook(hook.post(f"yolos")))

    elif level == 1:
        hooks.append(model.vit.register_forward_pre_hook(hook.pre(f"yolos.vit")))
        hooks.append(model.vit.register_forward_hook(hook.post(f"yolos.vit")))


    elif level == 2:
        hooks.append(model.vit.embeddings.register_forward_pre_hook(hook.pre(f"yolos.vit.embeddings")))
        hooks.append(model.vit.embeddings.register_forward_hook(hook.post(f"yolos.vit.embeddings")))

        hooks.append(model.vit.encoder.register_forward_pre_hook(hook.pre(f"yolos.vit.encoder")))
        hooks.append(model.vit.encoder.register_forward_hook(hook.post(f"yolos.vit.encoder")))

        hooks.append(model.vit.layernorm.register_forward_pre_hook(hook.pre(f"yolos.vit.layernorm")))
        hooks.append(model.vit.layernorm.register_forward_hook(hook.post(f"yolos.vit.layernorm")))

    elif level == 3:
        hooks.append(model.vit.embeddings.register_forward_pre_hook(hook.pre(f"yolos.vit.embeddings")))
        hooks.append(model.vit.embeddings.register_forward_hook(hook.post(f"yolos.vit.embeddings")))

        #Start encoder
        for i, layer in enumerate(model.vit.encoder.layer):
            hooks.append(layer.register_forward_pre_hook(hook.pre(f"yolos.vit.encoder.layer_{i}")))
            hooks.append(layer.register_forward_hook(hook.post(f"yolos.vit.encoder.layer_{i}")))


        hooks.append(model.vit.encoder.interpolation.register_forward_pre_hook(hook.pre(f"yolos.vit.encoder.interpolation")))
        hooks.append(model.vit.encoder.interpolation.register_forward_hook(hook.post(f"yolos.vit.encoder.interpolation")))
        #End encoder

        hooks.append(model.vit.layernorm.register_forward_pre_hook(hook.pre(f"yolos.vit.layernorm")))
        hooks.append(model.vit.layernorm.register_forward_hook(hook.post(f"yolos.vit.layernorm")))       

    if level != 0:
        hooks.append(model.class_labels_classifier.register_forward_pre_hook(hook.pre(f"yolos.class_labels_classifier")))
        hooks.append(model.class_labels_classifier.register_forward_hook(hook.post(f"yolos.class_labels_classifier")))

        hooks.append(model.bbox_predictor.register_forward_pre_hook(hook.pre(f"yolos.bbox_predictor")))
        hooks.append(model.bbox_predictor.register_forward_hook(hook.post(f"yolos.bbox_predictor")))


    with torch.no_grad():
        model(image)
        model(image)
        hook._record = []
        output = model(image)






    if level != 0:
        pred_box_size = output.pred_boxes.element_size() * output.pred_boxes.nelement() / 1024 / 1024
        logit_size = output.logits.element_size() * output.logits.nelement() / 1024 / 1024
        last_hidden_size = output.last_hidden_state.element_size() * output.last_hidden_state.nelement() / 1024 / 1024
        output_size = pred_box_size + logit_size + last_hidden_size
        hook._record.append({"layer_name" : "output", "time" : 0, "cpu_mem" : 0, "cuda_mem": output_size, "size" : output_size,  "MACs": 0})

    for h in hooks:
        h.remove()

    df = pd.DataFrame(hook.record)

    df.loc['Total'] = df.sum(numeric_only=True)

    # print(df.sum(numeric_only=True))

    df.loc[df.index[-1], 'layer_name'] = 'Total'

    print(df)

    df.to_csv(f"yolos_prof_{level}.csv", index=False)





class YoloS_Scaled(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-base").eval()
        self.config = self.model.config

    def forward(self, **input):
        return self.model(**input)
    

model = YoloS_Scaled()
#print(model.model.vit)
#print(model.model.config)

#test_yolos(model)

profile_yolo(0)
#test_yolos(model)