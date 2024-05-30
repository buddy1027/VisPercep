import gradio as gr
from lavis.models import load_model_and_preprocess
import torch
import argparse
from PIL import Image
from ram import get_transform, inference_ram
from ram.models import ram

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
ram_checkpoint = "/scratch/qo234/pretrained/spaces--xinyu1205--recognize-anything/snapshots/4b6c1161a44412eb4cf17633df36f5ad084dbec6/ram_swin_large_14m.pth"
tag2text_checkpoint = "/scratch/qo234/pretrained/spaces--xinyu1205--recognize-anything/snapshots/4b6c1161a44412eb4cf17633df36f5ad084dbec6/tag2text_swin_14m.pth"
image_size = 384


@torch.no_grad()
def inference_method(raw_image, specified_tags, tagging_model_type, tagging_model, transform):
    
    print(f"Start processing, image size {raw_image.size}")
    image = transform(raw_image).unsqueeze(0).to(device)
    if tagging_model_type == "RAM":
        res = inference_ram(image, tagging_model)
        tags = res[0].strip(' ').replace('  ', ' ')
        # tags_chinese = res[1].strip(' ').replace('  ', ' ')
        tags = tags.replace(" | ", ", ")
        print("Tags: ", tags)
        return tags


def inference_with_ram(img):
    return inference_method(img, None, "RAM", ram_model, transform)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model-name", default="blip2_vicuna_instruct")
    parser.add_argument("--model-type", default="vicuna7b")
    args = parser.parse_args()

    transform = get_transform(image_size=image_size)
    ram_model = ram(pretrained=ram_checkpoint, image_size=image_size, vit='swin_l').eval().to(device)

    image_input = gr.Image(type="pil")
    min_len = gr.Slider(
        minimum=1,
        maximum=50,
        value=1,
        step=1,
        interactive=True,
        label="Min Length",
    )

    max_len = gr.Slider(
        minimum=10,
        maximum=500,
        value=500,
        step=5,
        interactive=True,
        label="Max Length",
    )

    sampling = gr.Radio(
        choices=["Beam search", "Nucleus sampling"],
        value="Beam search",
        label="Text Decoding Method",
        interactive=True,
    )

    top_p = gr.Slider(
        minimum=0.5,
        maximum=1.0,
        value=0.9,
        step=0.1,
        interactive=True,
        label="Top p",
    )

    beam_size = gr.Slider(
        minimum=1,
        maximum=10,
        value=5,
        step=1,
        interactive=True,
        label="Beam Size",
    )

    len_penalty = gr.Slider(
        minimum=-1,
        maximum=2,
        value=1,
        step=0.2,
        interactive=True,
        label="Length Penalty",
    )

    repetition_penalty = gr.Slider(
        minimum=-1,
        maximum=3,
        value=3,
        step=0.2,
        interactive=True,
        label="Repetition Penalty",
    )

    prompt_textbox = gr.Textbox(label="Prompt:", placeholder="prompt", lines=2)
    print('Loading model...')
    model, vis_processors, _ = load_model_and_preprocess(
        name=args.model_name,
        model_type=args.model_type,
        is_eval=True,
        device=device,
    )
    print('Loading model done!')

    def inference(image, prompt, min_len, max_len, beam_size, len_penalty, repetition_penalty, top_p, decoding_method, modeltype):
        use_nucleus_sampling = decoding_method == "Nucleus sampling"
        print(image, prompt, min_len, max_len, beam_size, len_penalty, repetition_penalty, top_p, use_nucleus_sampling)
        # Get Tags
        ram_out_tag = inference_with_ram(image)
        # Get prompt
        tag_text = ram_out_tag
        prompt = prompt.format(tags=tag_text)

        print(prompt)
        image = vis_processors["eval"](image).unsqueeze(0).to(device)
        # Input for VLM
        samples = {
            "image": image,
            "prompt": prompt,
        }
        # Generate from VLM
        output = model.generate(
            samples,
            length_penalty=float(len_penalty),
            repetition_penalty=float(repetition_penalty),
            num_beams=beam_size,
            max_length=max_len,
            min_length=min_len,
            top_p=top_p,
            use_nucleus_sampling=use_nucleus_sampling,
        )
        return output[0]

    gr.Interface(
        fn=inference,
        inputs=[image_input, prompt_textbox, min_len, max_len, beam_size, len_penalty, repetition_penalty, top_p, sampling],
        outputs="text",
        allow_flagging="never",
    ).launch(share=True)
