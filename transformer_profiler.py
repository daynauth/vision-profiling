import torch
import pandas as pd

from detect import YoloS
from profiler import Profiler
from hook import TraceHook, ProfHook
from time import sleep

from torch.profiler import profile, record_function, ProfilerActivity

from transformers import AutoImageProcessor, AutoModelForObjectDetection





#model = YoloS().eval()


def generate_dep(model):
    hook = TraceHook()
    profiler = Profiler(model, "yolos", hook)

    image = torch.rand(1, 3, 416, 416)
    image.__tag__ = "input"

    record = profiler.run(image)
    df = pd.DataFrame(record)
    df.to_csv("dep.csv", index=False)


def generate_prof(model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    image = torch.rand(1, 3, 640, 640).to(device)


    hook = ProfHook()
    hook.add_record("input")

    profiler = Profiler(model, "yolos", hook)
    record = profiler.run(image)
    df = pd.DataFrame(record)
    df.to_csv("prof.csv", index=False)


def yolos_profiler(hook, filename="prof.csv", tag = False):
    hooks = []
    #model = YoloS().eval()
    model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-base").eval()
    #warmup

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    image = torch.rand(1, 3, 640, 640).to(device)   

    memory = torch.cuda.memory_allocated()
    print(f"memory usage: {memory/1024/1024}MB")

    # for _ in range(10):

    #     # memory = torch.cuda.memory_allocated()
    #     # print(f"memory usage: {memory/1024/1024}MB")

    #     # reserved = torch.cuda.memory_reserved()
    #     # print(f"reserved memory usage: {reserved/1024/1024}MB")
    #     memory_stats = torch.cuda.memory_stats()

    #     for key, value in memory_stats.items():
    #         if value > (1024 * 1024):
    #             print(f"{key} : {value/1024/1024}MB")


    #     print("--------------------------------------------------")

    #     with torch.no_grad():
    #         model(image)

    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], profile_memory=True, with_stack=True) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                model(image)



    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], profile_memory=True, with_stack=True) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                model(image)

    print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_memory_usage"))
    
    # max_mem = torch.cuda.max_memory_allocated()
    # max_reserved_mem = torch.cuda.max_memory_reserved()
    # print(f"max memory usage: {max_mem/1024/1024}MB")
    # print(f"max reserved memory usage: {max_reserved_mem/1024/1024}MB")
    # print(f"total memory usage: {(max_mem + max_reserved_mem)/1024/1024}MB")


    # prof.export_chrome_trace("trace.json")

    # if tag:
    #     image.__tag__ = "input"


    # # for _ in range(10):
    # #     model(image) 
    #     #sleep(5000)

    # '''
    # - confirm the model memory usage
    # - 
    # '''

    image_size = image.numel() * 4 / 1024 / 1024


    hooks.append(model.vit.embeddings.register_forward_pre_hook(hook.pre(f"embeddings")))
    hooks.append(model.vit.embeddings.register_forward_hook(hook.post(f"embeddings")))


    
    for i, layer in enumerate(model.vit.encoder.layer):
        profile_attention = False
        if profile_attention:
            profile_self_attention = False
            if profile_self_attention:
                #query layer hooks
                hooks.append(layer.attention.attention.query.register_forward_pre_hook(hook.pre(f"encoder_{i}_attention_self_attention_query")))
                hooks.append(layer.attention.attention.query.register_forward_hook(hook.post(f"encoder_{i}_attention_self_attention_query")))

                #key layer hooks
                hooks.append(layer.attention.attention.key.register_forward_pre_hook(hook.pre(f"encoder_{i}_attention_self_attention_key")))
                hooks.append(layer.attention.attention.key.register_forward_hook(hook.post(f"encoder_{i}_attention_self_attention_key")))

                #value layer hooks
                hooks.append(layer.attention.attention.value.register_forward_pre_hook(hook.pre(f"encoder_{i}_attention_self_attention_value")))
                hooks.append(layer.attention.attention.value.register_forward_hook(hook.post(f"encoder_{i}_attention_self_attention_value")))

                #dropout layer hooks
                hooks.append(layer.attention.attention.dropout.register_forward_pre_hook(hook.pre(f"encoder_{i}_attention_self_attention_dropout")))
                hooks.append(layer.attention.attention.dropout.register_forward_hook(hook.post(f"encoder_{i}_attention_self_attention_dropout")))
            else:
                #attention layers hooks
                hooks.append(layer.attention.attention.register_forward_pre_hook(hook.pre(f"encoder_{i}_attention_self_attention")))
                hooks.append(layer.attention.attention.register_forward_hook(hook.post(f"encoder_{i}_attention_self_attention")))

            profile_attention_output = False
            if profile_attention_output:
                hooks.append(layer.attention.output.dense.register_forward_pre_hook(hook.pre(f"encoder_{i}_attention_output_dense")))
                hooks.append(layer.attention.output.dense.register_forward_hook(hook.post(f"encoder_{i}_attention_output_dense")))

                hooks.append(layer.attention.output.dropout.register_forward_pre_hook(hook.pre(f"encoder_{i}_attention_output_dropout")))
                hooks.append(layer.attention.output.dropout.register_forward_hook(hook.post(f"encoder_{i}_attention_output_dropout")))
            else:
                #attention output layer hooks
                hooks.append(layer.attention.output.register_forward_pre_hook(hook.pre(f"encoder_{i}_attention_output")))
                hooks.append(layer.attention.output.register_forward_hook(hook.post(f"encoder_{i}_attention_output")))

        else:
            #attention layers hooks
            hooks.append(layer.attention.register_forward_pre_hook(hook.pre(f"encoder_{i}_attention")))
            hooks.append(layer.attention.register_forward_hook(hook.post(f"encoder_{i}_attention")))


        #intermediate layer hooks
        profile_intermediate = False
        if profile_intermediate:
            hooks.append(layer.intermediate.dense.register_forward_pre_hook(hook.pre(f"encoder_{i}_intermediate_dense")))
            hooks.append(layer.intermediate.dense.register_forward_hook(hook.post(f"encoder_{i}_intermediate_dense")))

            hooks.append(layer.intermediate.intermediate_act_fn.register_forward_pre_hook(hook.pre(f"encoder_{i}_intermediate_act_fn")))
            hooks.append(layer.intermediate.intermediate_act_fn.register_forward_hook(hook.post(f"encoder_{i}_intermediate_act_fn")))
        else:
            hooks.append(layer.intermediate.register_forward_pre_hook(hook.pre(f"encoder_{i}_intermediate")))
            hooks.append(layer.intermediate.register_forward_hook(hook.post(f"encoder_{i}_intermediate")))

        #output layer hooks
        profile_output = False
        if profile_output:
            hooks.append(layer.output.dense.register_forward_pre_hook(hook.pre(f"encoder_{i}_output_dense")))
            hooks.append(layer.output.dense.register_forward_hook(hook.post(f"encoder_{i}_output_dense")))

            hooks.append(layer.output.dropout.register_forward_pre_hook(hook.pre(f"encoder_{i}_output_dropout")))
            hooks.append(layer.output.dropout.register_forward_hook(hook.post(f"encoder_{i}_output_dropout")))
        else:
            hooks.append(layer.output.register_forward_pre_hook(hook.pre(f"encoder_{i}_output")))
            hooks.append(layer.output.register_forward_hook(hook.post(f"encoder_{i}_output")))

        hooks.append(layer.layernorm_before.register_forward_pre_hook(hook.pre(f"encoder_{i}_layernorm_before")))
        hooks.append(layer.layernorm_before.register_forward_hook(hook.post(f"encoder_{i}_layernorm_before")))

        hooks.append(layer.layernorm_after.register_forward_pre_hook(hook.pre(f"encoder_{i}_layernorm_after")))
        hooks.append(layer.layernorm_after.register_forward_hook(hook.post(f"encoder_{i}_layernorm_after")))


    hooks.append(model.vit.encoder.interpolation.register_forward_pre_hook(hook.pre(f"encoder_interpolation")))
    hooks.append(model.vit.encoder.interpolation.register_forward_hook(hook.post(f"encoder_interpolation")))


    hooks.append(model.vit.layernorm.register_forward_pre_hook(hook.pre(f"layer_norm")))
    hooks.append(model.vit.layernorm.register_forward_hook(hook.post(f"layer_norm")))

    hooks.append(model.class_labels_classifier.register_forward_pre_hook(hook.pre(f"class_labels_classifier")))
    hooks.append(model.class_labels_classifier.register_forward_hook(hook.post(f"class_labels_classifier")))

    hooks.append(model.bbox_predictor.register_forward_pre_hook(hook.pre(f"bbox_predictor")))
    hooks.append(model.bbox_predictor.register_forward_hook(hook.post(f"bbox_predictor")))

    


    with torch.no_grad():
        model(image)
        model(image)
        hook._record = []

        hook._record.append({"layer_name" : "input", "time" : 0, "cpu_mem" : 0, "cuda_mem": 0, "size" : image_size,  "MACs": 0})
        model(image)



    hook._record.append({"layer_name" : "output", "time" : 0, "cpu_mem" : 0, "cuda_mem": 0, "size" : 0,  "MACs": 0})

    for h in hooks:
        h.remove()


    df = pd.DataFrame(hook.record)
    df.to_csv(filename, index=False)


    df.loc['Total'] = df.sum(numeric_only=True)
    print(df)

    # print(model.embeddings.patch_embeddings)

    # max_mem = torch.cuda.max_memory_allocated()
    # print(f"max memory usage: {max_mem/1024/1024}MB")

    print(model)
    #print(model.config)

yolos_profiler(ProfHook(), "prof.csv")
