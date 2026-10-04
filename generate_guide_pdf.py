import os
import sys
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak, KeepTogether
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#64748b"))
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 11 * inch - 36, "SignTalk AI — Complete Developer & System Guide")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(54, 11 * inch - 42, 8.5 * inch - 54, 11 * inch - 42)
        # Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 54, 36, page_text)
        self.drawString(54, 36, "CONFIDENTIAL — Final Year Capstone & Research Project")
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(54, 48, 8.5 * inch - 54, 48)
        self.restoreState()

def create_guide_pdf(filename="SignTalk_AI_Developer_Guide.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom color palette
    c_primary = colors.HexColor("#0f172a") # Slate 900
    c_accent = colors.HexColor("#2563eb")  # Blue 600
    c_accent_dark = colors.HexColor("#1d4ed8")
    c_success = colors.HexColor("#059669")
    c_warning = colors.HexColor("#d97706")
    c_danger = colors.HexColor("#dc2626")
    c_bg_light = colors.HexColor("#f8fafc")
    c_border = colors.HexColor("#cbd5e1")
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=c_primary,
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=c_accent,
        spaceAfter=15
    )
    
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#475569"),
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'Heading1Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=c_primary,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=c_accent_dark,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6
    )
    
    bullet_style = ParagraphStyle(
        'BulletCustom',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )
    
    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#0f172a")
    )
    
    code_style = ParagraphStyle(
        'CodeText',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#0f172a")
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("SignTalk AI: Complete System & Developer Guide", title_style))
    story.append(Paragraph("Bidirectional AI Communication Bridge for Hearing & Speech Impaired", subtitle_style))
    story.append(Paragraph(
        "<b>Department:</b> Information Science & Engineering, RNS Institute of Technology<br/>"
        "<b>Academic Year:</b> 2025–2026 | <b>Batch:</b> #24 | <b>Guide:</b> Umesh M<br/>"
        "<b>Team:</b> Arjun B Shetty, Khushi Ramesh, Mahin S Kunder, Sathya Shriya G N<br/>"
        "<b>Status:</b> Production prototype live. Ready for final validation, tuning, and IEEE paper integration.",
        meta_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_accent, spaceBefore=4, spaceAfter=14))

    # Section 1: Executive Summary & System Goals
    story.append(Paragraph("1. Executive Summary & Problem Statement", h1_style))
    story.append(Paragraph(
        "Over 466 million people globally suffer from disabling hearing loss. Conventional communication tools either rely on specialized hardware or awkward back-and-forth typing. <b>SignTalk AI</b> eliminates this friction through a dual-channel multimodal AI platform:",
        body_style
    ))
    story.append(Paragraph("• <b>Sign Language to Spoken Voice:</b> Translates continuous Indian Sign Language (ISL) from a standard webcam into natural grammatical speech with facial emotion awareness.", bullet_style))
    story.append(Paragraph("• <b>Speech to Sign Subtitles:</b> Transcribes the hearing user's voice into real-time visual subtitles with Whisper AI.", bullet_style))
    story.append(Paragraph("• <b>Multilingual & Context Aware:</b> Translates dialogues between English, Hindi, and Kannada, backed by Gemini 2.0 Flash for sentence reconstruction.", bullet_style))
    story.append(Paragraph("• <b>Zero-Network Resilience (Airplane Mode):</b> Runs entirely on-device when connectivity drops, using quantized TFLite models, local Flan-T5, and offline phrasebooks.", bullet_style))

    story.append(Spacer(1, 8))

    # Section 2: End-to-End Pipeline Architecture
    story.append(Paragraph("2. End-to-End Pipeline Architecture", h1_style))
    story.append(Paragraph(
        "The system coordinates 6 modular machine learning stages across asynchronous WebSocket pipelines:",
        body_style
    ))

    arch_data = [
        [Paragraph("<b>Stage</b>", body_style), Paragraph("<b>Model / Library</b>", body_style), Paragraph("<b>Function & Specification</b>", body_style)],
        [
            Paragraph("<b>1. Pose Extraction</b>", body_style),
            Paragraph("MoveNet Thunder (TF Hub)", body_style),
            Paragraph("Extracts 17 keypoints (body joints, shoulders, wrists). Coordinates normalized relative to shoulder midpoint and bounding-box diagonal.", body_style)
        ],
        [
            Paragraph("<b>2. Sequence Classifier</b>", body_style),
            Paragraph("BiLSTM (Keras / TFLite)", body_style),
            Paragraph("Evaluates 30-frame rolling sequence (shape: 30x17x2). Trained on INCLUDE dataset (40 curated classes, 89.4% val accuracy). Debounced across 3 agreeing frames.", body_style)
        ],
        [
            Paragraph("<b>3. Emotion Detection</b>", body_style),
            Paragraph("DeepFace (7 classes)", body_style),
            Paragraph("Runs on every 10th frame to minimize CPU overhead. Classifies emotion: happy, sad, angry, fear, surprise, neutral, disgust.", body_style)
        ],
        [
            Paragraph("<b>4. NLP Sentence Fixer</b>", body_style),
            Paragraph("Gemini 2.0 Flash / Flan-T5", body_style),
            Paragraph("Fuses recognized gesture keywords + dominant emotion + conversation history into a natural grammatical sentence. Falls back to local Flan-T5.", body_style)
        ],
        [
            Paragraph("<b>5. Multilingual Translation</b>", body_style),
            Paragraph("Google Translate REST / Cache", body_style),
            Paragraph("Translates sentences to English, Hindi, or Kannada. Uses local LRU cache and offline phrasebook fallback.", body_style)
        ],
        [
            Paragraph("<b>6. Speech & Synthesis</b>", body_style),
            Paragraph("Whisper Small & gTTS / Coqui", body_style),
            Paragraph("Whisper Small transcribes hearing voice; gTTS/Coqui synthesizes translated text into spoken audio.", body_style)
        ]
    ]

    t_arch = Table(arch_data, colWidths=[1.4*inch, 1.8*inch, 3.8*inch])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_bg_light),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_arch)

    story.append(PageBreak())

    # Section 3: Critical Rules & Locked Contracts
    story.append(Paragraph("3. Locked Contracts (NEVER MODIFY WITHOUT FULL-TEAM REVIEW)", h1_style))
    story.append(Paragraph(
        "These contracts link the frontend, backend, ML models, and mobile app together. Changing function arguments or array dimensions will break other layers:",
        body_style
    ))

    contracts = [
        "<b>1. Exactly 17 Keypoints (MoveNet Thunder):</b> The IEEE paper mentions 33 keypoints (MediaPipe), but our entire production ML pipeline is locked to <b>17 keypoints</b>. Do NOT switch to MediaPipe without retraining everything.",
        "<b>2. Classifier Signature:</b> <code>classify_sequence(sequence: np.ndarray)</code> takes exactly shape <code>(30, 17, 2)</code> and returns <code>{'label': str, 'confidence': float}</code>.",
        "<b>3. WebSocket Protocol (/ws/gesture):</b> Client streams JPEG frames at 20–30 FPS. Backend stabilizes output: requires <b>3 consecutive agreeing predictions</b> and a <b>1.5s cooldown</b> before re-emitting the same sign.",
        "<b>4. Unmirrored Webcam Rule:</b> In <code>frontend/src/components/WebcamView.jsx</code>, <code>mirrored={true}</code> must remain disabled. Mirroring flips wrists horizontally, which ruins recognition against the unmirrored training dataset.",
        "<b>5. Frame Skip Setting (?frame_skip=2):</b> Live recognition WebSocket must connect with <code>?frame_skip=2</code>. Because real signs take ~2.5–3s, at 20 FPS a 30-frame window with <code>skip=2</code> spans ~3.0s, matching the training clip duration.",
        "<b>6. Shared Offline Mode:</b> All services check <code>X-Offline-Mode: true</code> header or <code>?offline=true</code> query param through <code>api.core.offline_mode.get_offline_mode</code>."
    ]
    for c in contracts:
        story.append(Paragraph(f"• {c}", bullet_style))

    story.append(Spacer(1, 10))

    # Section 4: Repository Layout & Key Files
    story.append(Paragraph("4. Repository Structure & File Guide", h1_style))
    repo_data = [
        [Paragraph("<b>Directory / File</b>", body_style), Paragraph("<b>Role & Responsibility</b>", body_style)],
        [Paragraph("<code>backend/api/</code>", code_style), Paragraph("Production FastAPI backend: routes for pose, emotion, AI, speech, translation, analytics.", body_style)],
        [Paragraph("<code>backend/keypoint_utils.py</code>", code_style), Paragraph("MoveNet extraction, spatial normalization, temporal smoothing buffer.", body_style)],
        [Paragraph("<code>backend/classify.py</code>", code_style), Paragraph("Central gesture classification logic loading the active BiLSTM model.", body_style)],
        [Paragraph("<code>backend/nlp_correction.py</code>", code_style), Paragraph("Gemini 2.0 Flash prompt runner with local Flan-T5-small fallback.", body_style)],
        [Paragraph("<code>backend/emotion.py</code>", code_style), Paragraph("DeepFace facial emotion recognition and 2-second majority voting filter.", body_style)],
        [Paragraph("<code>backend/translation.py</code>", code_style), Paragraph("Translation engine: in-memory cache -> Firestore -> Google API -> phrasebook.", body_style)],
        [Paragraph("<code>backend/train_bilstm.py</code>", code_style), Paragraph("BiLSTM training script with stratified train/val split and TFLite exporter.", body_style)],
        [Paragraph("<code>backend/runs/</code>", code_style), Paragraph("Trained model weights (active: <code>exp_top40_stratified</code>, 89.4% accuracy).", body_style)],
        [Paragraph("<code>frontend/src/</code>", code_style), Paragraph("React web dashboard: WebcamView, SubtitleBar, LanguageSelector, AnalyticsPanel.", body_style)],
        [Paragraph("<code>mobile/</code>", code_style), Paragraph("Flutter app (cross-platform client sharing the same WebSocket/REST contracts).", body_style)]
    ]
    t_repo = Table(repo_data, colWidths=[2.2*inch, 4.8*inch])
    t_repo.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_bg_light),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_repo)

    story.append(Spacer(1, 10))

    # Section 5: Local Setup & Running
    story.append(Paragraph("5. Step-by-Step Developer Quickstart", h1_style))
    story.append(Paragraph("<b>Terminal 1 — Backend (FastAPI + WebSocket on Port 8000):</b>", body_style))
    story.append(Paragraph("<code>cd backend<br/>.\\venv\\Scripts\\activate<br/>uvicorn api.socket_manager:socket_app --host 127.0.0.1 --port 8000 --reload</code>", code_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Terminal 2 — Frontend (Vite React on Port 3000):</b>", body_style))
    story.append(Paragraph("<code>cd frontend<br/>npm run dev</code>", code_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("• Frontend URL: <b>http://localhost:3000</b><br/>• Swagger Docs: <b>http://127.0.0.1:8000/docs</b>", meta_style))

    story.append(PageBreak())

    # Section 6: Known Gotchas & Priority Fixes
    story.append(Paragraph("6. Immediate Gotchas & Action Items for the Week", h1_style))
    story.append(Paragraph(
        "To finish the project within the next few days, focus exclusively on these key fixes:",
        body_style
    ))

    tasks = [
        ("1. TFHub MoveNet Cache Corruption", "HIGH", "If Uvicorn crashes at startup saying 'saved_model.pb not found', clear the directory %TEMP%\\tfhub_modules. The script will automatically re-download a fresh copy."),
        ("2. Cloud Translation API Key & Billing", "HIGH", "Google Cloud Translation API requires billing. If inactive, the system gracefully falls back to phrasebook.json. Either provide a billing-enabled key or expand phrasebook.json for demo phrases."),
        ("3. Firebase Service Account & Mock Mode", "MEDIUM", "Ensure backend/secrets/firebase-service-account.json is present. If testing without GCP credentials, verify that the firebase client's mock fallback is enabled so Auth and Firestore calls don't block login."),
        ("4. Fine-Tuning Sentence Correction Prompts", "MEDIUM", "Check backend/nlp_correction.py prompt templates. Ensure Gemini returns concise sentences that sound natural and preserve the emotional tone."),
        ("5. Latency & Demo Preparation", "HIGH", "Verify that end-to-end latency remains under 300ms. Prepare 5 standard demo gestures from the 40-word vocabulary (e.g. HELLO, THANK YOU, PLEASE, HELP, GOOD) with clear lighting.")
    ]

    task_data = [[Paragraph("<b>Priority Item</b>", body_style), Paragraph("<b>Severity</b>", body_style), Paragraph("<b>Action Required</b>", body_style)]]
    for title, sev, desc in tasks:
        badge_color = c_danger if sev == "HIGH" else c_warning
        task_data.append([
            Paragraph(f"<b>{title}</b>", body_style),
            Paragraph(f"<font color='{badge_color}'><b>{sev}</b></font>", body_style),
            Paragraph(desc, body_style)
        ])

    t_task = Table(task_data, colWidths=[2.2*inch, 0.9*inch, 3.9*inch])
    t_task.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_bg_light),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_task)

    story.append(Spacer(1, 14))

    # Section 7: Dataset & Model Training Summary
    story.append(Paragraph("7. Datasets & Model Training Reference", h1_style))
    story.append(Paragraph(
        "• <b>Dataset:</b> INCLUDE (Indian Sign Language dataset). All 15 categories preprocessed into 4,257 sequence files (<code>backend/data/sequences/</code>).<br/>"
        "• <b>Curated Vocabulary:</b> 40 top-represented classes with stratified train/validation split achieving <b>89.4% accuracy</b>.<br/>"
        "• <b>Training Command:</b> <code>python train_bilstm.py --data_dir data/sequences --labels_csv data/labels_top40.csv --split_mode stratified --output_dir runs/exp_top40_stratified</code><br/>"
        "• <b>TFLite Conversion:</b> <code>python convert_to_tflite.py --saved_model runs/exp_top40_stratified/saved_model --output models/bilstm.tflite</code>",
        body_style
    ))

    story.append(Spacer(1, 14))

    # Callout Box
    story.append(Table(
        [[Paragraph("<b>Pro-Tip for Handover:</b> Have your friend start with Section 5 (Quickstart), run both terminals, and test the Swagger UI at http://127.0.0.1:8000/docs. The system is modular: frontend UI, gesture pipeline, and NLP correction can each be debugged and modified independently.", callout_style)]],
        colWidths=[7.0*inch],
        style=[
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#eff6ff")),
            ('BOX', (0,0), (-1,-1), 1, c_accent),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ]
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {filename}")

if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "SignTalk_AI_Developer_Guide.pdf"
    create_guide_pdf(out_file)
