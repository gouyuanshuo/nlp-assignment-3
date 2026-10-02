"""Local upload interface; classifier errors always clear previous results."""

import logging
from pathlib import Path


def empty_result():
    return ("Upload a short English video to begin.", None, "", [])


def analyze_upload(video, classifier, progress=None):
    yield ("Analyzing your video…", *empty_result()[1:])
    if not video:
        yield ("Choose a video first.", *empty_result()[1:])
        return
    try:
        result = classifier.classify(Path(video), progress=progress)
        rows = [
            [
                w["start"],
                w["end"],
                w["prediction"],
                max(w["prob_negative"], w["prob_positive"]),
                w["text"],
            ]
            for w in result["windows"]
        ]
        status = f"{result['prediction'].capitalize()} sentiment · analyzed in {result['processing_seconds']:.1f} seconds"
        if result.get("smoke_checkpoint"):
            status += " · integration-test checkpoint; final Colab training pending"
        yield (status, result["probabilities"], result["transcript"], rows)
    except Exception as exc:
        logging.getLogger(__name__).exception("Video analysis failed")
        message = (
            str(exc)
            if isinstance(exc, (ValueError, FileNotFoundError))
            else "Analysis failed. See the local server log for details."
        )
        yield (message, *empty_result()[1:])


def build_demo(classifier):
    import gradio as gr

    with gr.Blocks(
        title="Video Sentiment · NLP Assignment 3",
        theme=gr.themes.Soft(primary_hue="teal"),
        delete_cache=(60, 3600),
    ) as demo:
        gr.Markdown(
            "# Video sentiment\nSee how speech, voice and facial expression combine into a sentiment prediction."
        )
        gr.Markdown(
            "Upload an English MP4, MOV or WebM: **up to 60 seconds and 200 MB**, with one visible speaker. This research model classifies **positive or negative** sentiment; neutral was excluded from training."
        )
        with gr.Row():
            with gr.Column():
                video = gr.Video(label="Your video", sources=["upload"])
                analyze = gr.Button("Analyze video", variant="primary")
            with gr.Column():
                status = gr.Textbox(label="Status", value=empty_result()[0], interactive=False)
                probabilities = gr.Label(label="Sentiment probabilities", num_top_classes=2)
                transcript = gr.Textbox(label="Recognized speech", lines=6, interactive=False)
        windows = gr.Dataframe(
            headers=["Start (s)", "End (s)", "Sentiment", "Probability", "Speech"],
            datatype=["number", "number", "str", "number", "str"],
            label="Prediction by speech window",
            interactive=False,
        )
        gr.Markdown(
            "Processed on this computer. Temporary analysis files are removed after each request; the upload preview is automatically deleted within about an hour."
        )
        outputs = [status, probabilities, transcript, windows]

        def process(path, progress=gr.Progress()):
            for result in analyze_upload(
                path, classifier, lambda value, message: progress(value, desc=message)
            ):
                yield (*result, gr.update(interactive=False), gr.update(interactive=False))

        event = analyze.click(
            process,
            video,
            outputs + [video, analyze],
            concurrency_limit=1,
            api_name="analyze_video",
        )
        event.then(
            lambda: (gr.update(interactive=True), gr.update(interactive=True)),
            outputs=[video, analyze],
            api_name=False,
        )
        video.change(empty_result, outputs=outputs, queue=False, api_name=False)
    return demo.queue(max_size=8)
