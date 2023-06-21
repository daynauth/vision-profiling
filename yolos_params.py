import json
import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoImageProcessor, AutoModelForObjectDetection



image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-base")
model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-base").eval()

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.shape)



# sum embeddings params
embeddings_params = {name: param.numel() for name, param in model.vit.embeddings.named_parameters() }
embedings_sum = {"Embeddings" : sum(embeddings_params.values())}


# sum Position encoder params
position_encoding_params = {"Interpolation": model.vit.encoder.mid_position_embeddings.numel()}

# sum encoder layer params
layer_params = {name: param.numel() for name, param in model.vit.encoder.layer.named_parameters() }
out = {f"Encoder.Layer_{i}": sum([value for key, value in layer_params.items() if key.startswith(str(i) + '.')]) for i in range(model.config.num_attention_heads)}

#layer norm params
layer_norm_sum = {"Layer Norm" :sum([param.numel() for name, param in model.vit.layernorm.named_parameters() ])}


# sum head params
bbox_predictor_sum  = {"Bbox_Predictor" : sum([param.numel() for _, param in model.bbox_predictor.named_parameters()])}


# sum class_predictor params
class_predictor_params = {"Classifier" : sum([param.numel() for name, param in model.class_labels_classifier.named_parameters()])}


model_params = {**embedings_sum, **position_encoding_params, **out, **layer_norm_sum, **bbox_predictor_sum, **class_predictor_params}
df = pd.DataFrame.from_dict(model_params, orient='index', columns=['Params'])
df.reset_index(inplace=True)
df = df.rename(columns={'index': 'Layer'})




#print param total in millions
df['Params'] = df['Params'].apply(lambda x: x/1000000)

df.to_csv("YOLOS_Params.csv", index=False)



total_params = df['Params'].sum()
df['Params %'] = df['Params'].apply(lambda x: round(x/total_params * 100, 2))
df = df.reindex(df.index[::-1])

print(df)

#plot bar chart
ax = df.plot.barh(x = 'Layer', y = 'Params %', figsize=(10, 10), title="YOLOS Params", xlabel="Params in Millions", ylabel="Layer")

ax.bar_label(ax.containers[0] ,fmt=' %.2f%%')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
#ax.spines['left'].set_visible(False)

ax.set(xlabel=None)
ax.set_title("YOLOS Parameters Percentage", fontsize=20)
ax.set_ylabel("Layer", fontsize=15)
ax.get_xaxis().set_ticks([])

plt.tight_layout()
plt.savefig("YOLOS_Params.png")