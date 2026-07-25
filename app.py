import io
import csv
import os
from datetime import datetime
import gradio as gr
import moondream as md


def log_row(filename, row_dict):
    """Append a row to a local CSV, creating it with headers if it doesn't exist yet."""
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def pil_to_bytes(pil_image):
    """Convert a PIL Image to PNG bytes so Ollama's client can accept it."""
    buf = io.BytesIO()
    pil_image.save(buf, format='PNG')
    return buf.getvalue()


DANGER_WARNING = (
    "⚠️ WARNING: This shows signs of serious electrical damage. "
    "Do not attempt to repair this yourself. Turn off power at the "
    "breaker and contact a licensed electrician immediately."
)

DANGER_KEYWORDS = [
    "spark", "sparking", "sparked", "smoke", "smoking", "burn", "burning",
    "burnt", "melt", "melted", "melting", "crack", "cracking", "fire",
    "flame", "shock", "shocked", "shocking", "arc", "arcing", "hot to touch",
    "buzzing", "smell", "smells", "smelled",
]

model = md.vl(api_key=os.environ.get("MOONDREAM_API_KEY"))

def description_flags_danger(description):
    """Cheap, deterministic check — catches danger signals the user already
    told us about in words, independent of whatever the vision model sees."""
    if not description:
        return False
    text = description.lower()
    return any(word in text for word in DANGER_KEYWORDS)


def check_danger(image):
    """Narrow, single-purpose check: does the photo show serious physical damage?"""
    danger_prompt = (
        "Look ONLY at physical damage in this image. Answer YES only if you "
        "clearly see melted or bubbled plastic, wiring pulled loose from its "
        "box, exposed bare copper twisted loose and unsecured, scorch or soot "
        "marks from actual fire, or visible burn/smoke damage.\n"
        "Answer NO for anything else, including: a lit indicator light, a "
        "tripped breaker switch, a flickering bulb, normal wear, or a circuit "
        "breaker panel with no visible melting or burning.\n"
        "Answer with exactly one word: YES or NO."
    )
    result = model.query(image, danger_prompt)
    answer = result["answer"].strip().upper()
    return answer.startswith('YES')


def diagnose_electrical(image, email, description):
    if image is None:
        return "Please upload a photo of the electrical issue."
    if not email or "@" not in email:
        return "Please enter a valid email so we can follow up for feedback — this is a soft-launch test and your input genuinely shapes what we build next."

    try:
        is_danger = description_flags_danger(description) or check_danger(image)

        log_row('signups.csv', {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'email': email,
            'description': description or "",
            'flagged_danger': is_danger,
        })

        if is_danger:
            return DANGER_WARNING

        prompt = (
            "You are an expert electrician. Look at the photo and description "
            "and give a normal, practical Diagnosis, numbered Fix steps, and a "
            "brief Why explanation."
        )
        result = model.query(image, prompt + "\nDescription: " + (description or ""))
        disclaimer = (
            "\n\n_Note: This is an AI-generated assessment, not a substitute "
            "for a licensed electrician. If you notice melting, burning, "
            "exposed wiring, or anything that looks or smells unusual, stop "
            "and contact a professional._"
        )
        return result["answer"] + disclaimer
    except Exception as e:
        return (
            "Something went wrong talking to the AI model. "
            "Make sure Ollama is running and the 'moondream' model is pulled "
            f"(run: ollama pull moondream).\n\nError detail: {e}"
        )


with gr.Blocks(title="Journeyman") as demo:
    gr.Markdown("# Journeyman\nAI that helps skilled workers work faster and safer")

    with gr.Tabs():
        with gr.Tab("AI Diagnosis"):
            gr.Markdown("Upload photo + describe")
            email = gr.Textbox(label="Email", placeholder="you@example.com")
            image = gr.Image(type="pil", label="Photo")
            text = gr.Textbox(label="Description", placeholder="Flickering lights or sparking outlet")
            output = gr.Textbox(label="Diagnosis & Steps")
            analyze_btn = gr.Button("Analyze")
            gr.Markdown("_Analysis may take up to 2 minutes on this demo — thanks for your patience._")
            analyze_btn.click(
                diagnose_electrical,
                [image, email, text],
                output,
                show_progress="full",
            )

            gr.Markdown("---\n**Was this helpful?** Your feedback directly shapes what we build next.")
            feedback_text = gr.Textbox(label="Feedback (what worked, what didn't, anything confusing)")
            feedback_btn = gr.Button("Submit Feedback")
            feedback_output = gr.Textbox(label="", show_label=False, interactive=False)

            def submit_feedback(email_val, feedback_val):
                if not feedback_val:
                    return "Please write a quick note before submitting."
                log_row('feedback.csv', {
                    'timestamp': datetime.now().isoformat(timespec='seconds'),
                    'email': email_val or "",
                    'feedback': feedback_val,
                })
                return "Thanks — really appreciate the feedback!"

            feedback_btn.click(submit_feedback, [email, feedback_text], feedback_output)

        with gr.Tab("Connect with a Worker (Coming Soon)"):
            gr.Markdown(
                """
                ## Connect with a Worker
                **Coming soon**
                Some problems go beyond what one person — or one AI — can solve alone.
                This is where you'll be able to reach another worker on your team, or a
                trusted peer in the field, to talk through a tough job in real time.
                - Ask a quick question when you're stuck on-site
                - Share a photo or description with a teammate instantly
                - Escalate automatically when the AI's confidence is low
                We're validating this with real tradespeople now — reach out if you'd
                like early access.
                """
            )
            name_input = gr.Textbox(label="Your name (optional)", placeholder="Jane Doe")
            interest_input = gr.Textbox(
                label="What kind of help would you want from a teammate?",
                placeholder="e.g. a second opinion on a tricky panel job"
            )
            notify_btn = gr.Button("Notify me when this launches")
            notify_output = gr.Textbox(label="", show_label=False, interactive=False)

            def notify_signup_1(name, interest):
                log_row('feature_interest.csv', {
                    'timestamp': datetime.now().isoformat(timespec='seconds'),
                    'feature': 'connect_with_worker',
                    'name': name or "",
                    'interest': interest or "",
                })
                return "Thanks — we'll be in touch when this feature is ready!"

            notify_btn.click(notify_signup_1, [name_input, interest_input], notify_output)

        with gr.Tab("Find a Certified Worker (Coming Soon)"):
            gr.Markdown(
                """
                ## Find a Certified Worker
                **Coming soon**
                After a diagnosis, homeowners and companies will be able to get
                matched directly with a certified worker registered on the
                platform — someone qualified to handle exactly the problem
                the AI just identified.
                - Diagnosis flows straight into a referral, no separate search
                - Workers are verified/certified before they can be matched
                - Built for both individual homeowners and companies needing
                  reliable, vetted help
                We're validating this with certified workers and homeowners now —
                reach out if you'd like early access.
                """
            )
            role_input = gr.Radio(
                ["Homeowner", "Company", "Certified Worker (want to register)"],
                label="I am a...",
            )
            interest_input2 = gr.Textbox(
                label="What would you want from this?",
                placeholder="e.g. a trusted electrician for a home rewiring job"
            )
            notify_btn2 = gr.Button("Notify me when this launches")
            notify_output2 = gr.Textbox(label="", show_label=False, interactive=False)

            def notify_signup_2(role, interest):
                log_row('feature_interest.csv', {
                    'timestamp': datetime.now().isoformat(timespec='seconds'),
                    'feature': 'certified_worker_referral',
                    'role': role or "",
                    'interest': interest or "",
                })
                return "Thanks — we'll be in touch when this feature is ready!"

            notify_btn2.click(notify_signup_2, [role_input, interest_input2], notify_output2)

demo.launch()
