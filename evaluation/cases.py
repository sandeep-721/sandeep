from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    query: str
    expected_sources: tuple[str, ...]
    expected_symbols: tuple[str, ...] = ()
    software: str | None = None
    project: str | None = None
    category: str = "general"


EVALUATION_CASES = (
    # ==============================================================
    # EXACT SYMBOL / CLASS
    # ==============================================================

    EvaluationCase(
        name="facial_emotion_controller_exact",
        query="FacialEmotionController",
        expected_sources=("FacialEmotionController.cs",),
        expected_symbols=("FacialEmotionController",),
        software="Unity",
        project="Facial_Test",
        category="exact_symbol",
    ),

    EvaluationCase(
        name="facial_phoneme_driver_exact",
        query="FacialPhonemeBlendShapeDriver",
        expected_sources=("FacialPhonemeBlendShapeDriver.cs",),
        expected_symbols=("FacialPhonemeBlendShapeDriver",),
        software="Unity",
        project="Facial_Test",
        category="exact_symbol",
    ),

    EvaluationCase(
        name="bone_driver_exact",
        query="BoneDriver",
        expected_sources=("BoneDriver.cs",),
        expected_symbols=("BoneDriver",),
        software="Unity",
        project="Facial_Test",
        category="exact_symbol",
    ),

    EvaluationCase(
        name="facial_jaw_controller_exact",
        query="FacialJawController",
        expected_sources=("FacialJawController.cs",),
        expected_symbols=("FacialJawController",),
        software="Unity",
        project="Facial_Test",
        category="exact_symbol",
    ),

    EvaluationCase(
        name="facial_eye_controller_exact",
        query="FacialEyeController",
        expected_sources=("FacialEyeController.cs",),
        expected_symbols=("FacialEyeController",),
        software="Unity",
        project="Facial_Test",
        category="exact_symbol",
    ),

    # ==============================================================
    # METHOD / FUNCTION RETRIEVAL
    # ==============================================================

    EvaluationCase(
        name="emotion_estimation_method",
        query="How is emotion estimated?",
        expected_sources=("FacialEmotionController.cs",),
        expected_symbols=("EstimateEmotion",),
        software="Unity",
        project="Facial_Test",
        category="method",
    ),

    EvaluationCase(
        name="emotion_prosody_method",
        query="How does the emotion system update prosody?",
        expected_sources=("FacialEmotionController.cs",),
        expected_symbols=("UpdateProsody",),
        software="Unity",
        project="Facial_Test",
        category="method",
    ),

    EvaluationCase(
        name="emotion_blendshape_mapping_method",
        query="How are emotion blend shapes automatically mapped?",
        expected_sources=("FacialEmotionController.cs",),
        expected_symbols=("AutoMapEmotionBlendShapes",),
        software="Unity",
        project="Facial_Test",
        category="method",
    ),

    EvaluationCase(
        name="bone_transform_lookup_method",
        query="How does BoneDriver find a transform in the hierarchy?",
        expected_sources=("BoneDriver.cs",),
        expected_symbols=("FindTransformInThisHierarchy",),
        software="Unity",
        project="Facial_Test",
        category="method",
    ),

    EvaluationCase(
        name="bone_update_method",
        query="How does BoneDriver update bone-driven animation?",
        expected_sources=("BoneDriver.cs",),
        expected_symbols=("LateUpdateBoneDriver",),
        software="Unity",
        project="Facial_Test",
        category="method",
    ),

    # ==============================================================
    # NATURAL LANGUAGE CODE QUESTIONS
    # ==============================================================

    EvaluationCase(
        name="facial_emotion_generation",
        query="How does FacialEmotionController generate facial emotions?",
        expected_sources=("FacialEmotionController.cs",),
        expected_symbols=("FacialEmotionController",),
        software="Unity",
        project="Facial_Test",
        category="natural_language_code",
    ),

    EvaluationCase(
        name="facial_phoneme_driver_behavior",
        query="How does the facial phoneme blend shape driver work?",
        expected_sources=("FacialPhonemeBlendShapeDriver.cs",),
        expected_symbols=("FacialPhonemeBlendShapeDriver",),
        software="Unity",
        project="Facial_Test",
        category="natural_language_code",
    ),

    EvaluationCase(
        name="bone_driver_behavior",
        query="What does BoneDriver control?",
        expected_sources=("BoneDriver.cs",),
        expected_symbols=("BoneDriver",),
        software="Unity",
        project="Facial_Test",
        category="natural_language_code",
    ),

    EvaluationCase(
        name="jaw_system_behavior",
        query="How is jaw movement controlled in the facial system?",
        expected_sources=("FacialJawController.cs",),
        expected_symbols=("FacialJawController",),
        software="Unity",
        project="Facial_Test",
        category="natural_language_code",
    ),

    EvaluationCase(
        name="eye_system_behavior",
        query="How does the facial eye controller handle eye movement?",
        expected_sources=("FacialEyeController.cs",),
        expected_symbols=("FacialEyeController",),
        software="Unity",
        project="Facial_Test",
        category="natural_language_code",
    ),

    # ==============================================================
    # LIP-SYNC / FACIAL PIPELINE
    # ==============================================================

    EvaluationCase(
        name="lip_sync_window",
        query="Where is the facial lip sync editor window implemented?",
        expected_sources=("FacialLipSyncWindow.cs",),
        software="Unity",
        project="Facial_Test",
        category="lip_sync",
    ),

    EvaluationCase(
        name="lip_sync_panel",
        query="Where is the facial lip sync panel implemented?",
        expected_sources=("FacialLipSyncPanel.cs",),
        software="Unity",
        project="Facial_Test",
        category="lip_sync",
    ),

    EvaluationCase(
        name="phoneme_audio_cutter",
        query="Where is the tool for cutting facial phoneme audio samples?",
        expected_sources=("FacialPhonemeAudioCutterWindow.cs",),
        software="Unity",
        project="Facial_Test",
        category="lip_sync",
    ),

    EvaluationCase(
        name="phoneme_calibration_recorder",
        query="How does the facial phoneme calibration recorder work?",
        expected_sources=("FacialPhonemeCalibrationRecorder.cs",),
        software="Unity",
        project="Facial_Test",
        category="lip_sync",
    ),

    EvaluationCase(
        name="phoneme_calibration_profile",
        query="Where is the facial phoneme calibration profile defined?",
        expected_sources=("FacialPhonemeCalibrationProfile.cs",),
        software="Unity",
        project="Facial_Test",
        category="lip_sync",
    ),

    # ==============================================================
    # PROJECT DOCUMENTATION
    # ==============================================================

    EvaluationCase(
        name="project_goal",
        query="What is the goal of the facial automation project?",
        expected_sources=("Project Goal.md",),
        software="Unity",
        project="Facial_Test",
        category="documentation",
    ),

    EvaluationCase(
        name="project_structure",
        query="What is the structure of the facial automation project?",
        expected_sources=("STRUCTURE_PLAN.md",),
        software="Unity",
        project="Facial_Test",
        category="documentation",
    ),

    EvaluationCase(
        name="master_architecture",
        query="What is the master architecture of the facial automation system?",
        expected_sources=("MASTER_ARCHITECTURE_PLAN.md",),
        software="Unity",
        project="Facial_Test",
        category="documentation",
    ),

    EvaluationCase(
        name="execution_plan",
        query="What is the execution plan for the facial automation project?",
        expected_sources=("EXECUTION_PLAN.md",),
        software="Unity",
        project="Facial_Test",
        category="documentation",
    ),

    EvaluationCase(
        name="project_progress",
        query="What is the current progress of the facial automation project?",
        expected_sources=("ProjectProgress.md",),
        software="Unity",
        project="Facial_Test",
        category="documentation",
    ),

    EvaluationCase(
        name="facial_project_capsule",
        query="What are the planned capabilities of the facial automation system?",
        expected_sources=("FacialAutomation_Project_Capsule.md",),
        software="Unity",
        project="Facial_Test",
        category="documentation",
    ),

    EvaluationCase(
        name="natural_lipsync_reference",
        query="What are the requirements for natural facial lip sync and speaking behavior?",
        expected_sources=(
            "Reference_Video_Analysis_and_Natural_LipSync.md",
        ),
        software="Unity",
        project="Facial_Test",
        category="documentation",
    ),

    # ==============================================================
    # CONFIGURATION / PROJECT METADATA
    # ==============================================================

    EvaluationCase(
        name="unity_project_version",
        query="What Unity version is this project using?",
        expected_sources=("ProjectVersion.txt",),
        software="Unity",
        project="Facial_Test",
        category="configuration",
    ),

    EvaluationCase(
        name="unity_package_manifest",
        query="Where is the Unity package manifest defined?",
        expected_sources=("manifest.json",),
        software="Unity",
        project="Facial_Test",
        category="configuration",
    ),

    EvaluationCase(
        name="facial_opencode_configuration",
        query="Where is the OpenCode configuration for the facial automation project?",
        expected_sources=("opencode.json",),
        software="Unity",
        project="Facial_Test",
        category="configuration",
    ),

    # ==============================================================
    # CROSS-FILE / RELATED COMPONENTS
    # ==============================================================

    EvaluationCase(
        name="lip_sync_editor_components",
        query="What files implement the facial lip sync editor workflow?",
        expected_sources=(
            "FacialLipSyncWindow.cs",
            "FacialLipSyncPanel.cs",
            "FacialPhonemeAudioCutterWindow.cs",
            "FacialPhonemeCalibrationRecorderEditor.cs",
        ),
        software="Unity",
        project="Facial_Test",
        category="cross_file",
    ),

    EvaluationCase(
        name="facial_audio_analysis_components",
        query="Which components analyze facial audio and phoneme information?",
        expected_sources=(
            "FacialAudioPhonemeAnalyzer.cs",
            "FacialFormantAnalyzer.cs",
            "FacialMfccAnalyzer.cs",
            "FacialIntrinsicVowelClassifier.cs",
        ),
        software="Unity",
        project="Facial_Test",
        category="cross_file",
    ),

    EvaluationCase(
        name="facial_control_components",
        query="Which components control jaw, eyes, head and facial output?",
        expected_sources=(
            "FacialJawController.cs",
            "FacialEyeController.cs",
            "FacialHeadAnimationController.cs",
            "FacialControlMixer.cs",
        ),
        software="Unity",
        project="Facial_Test",
        category="cross_file",
    ),

    # ==============================================================
    # REALLUSION / SECOND SOFTWARE-SUBSYSTEM
    # ==============================================================

    EvaluationCase(
        name="reallusion_bone_driver",
        query="Which Reallusion runtime file contains BoneDriver?",
        expected_sources=("BoneDriver.cs",),
        expected_symbols=("BoneDriver",),
        software="Unity",
        project="Facial_Test",
        category="cross_software_subsystem",
    ),

    EvaluationCase(
        name="reallusion_camera_proxy",
        query="How is the Reallusion camera proxy implemented?",
        expected_sources=("CameraProxy.cs",),
        software="Unity",
        project="Facial_Test",
        category="cross_software_subsystem",
    ),

    EvaluationCase(
        name="reallusion_wrinkle_manager",
        query="Where is the Reallusion wrinkle manager implemented?",
        expected_sources=("WrinkleManager.cs",),
        software="Unity",
        project="Facial_Test",
        category="cross_software_subsystem",
    ),

    # ==============================================================
    # UNIVERSAL RAG SELF-RETRIEVAL
    # ==============================================================

    EvaluationCase(
        name="rag_embedder",
        query="How does the RAG embedding component generate embeddings?",
        expected_sources=("embedder.py",),
        category="rag_self",
    ),

    EvaluationCase(
        name="rag_hybrid_search",
        query="How does the RAG hybrid search combine retrieval methods?",
        expected_sources=("hybrid_search.py",),
        category="rag_self",
    ),

    EvaluationCase(
        name="rag_reranker",
        query="How does the RAG reranker combine ranking signals?",
        expected_sources=("reranker.py",),
        category="rag_self",
    ),

    EvaluationCase(
        name="rag_sparse_search",
        query="How does the sparse search system perform lexical retrieval?",
        expected_sources=("sparse_search.py",),
        category="rag_self",
    ),

    EvaluationCase(
        name="rag_indexer",
        query="How does the RAG indexer process and index project files?",
        expected_sources=("indexer.py",),
        category="rag_self",
    ),

    EvaluationCase(
        name="rag_metadata_schema",
        query="Where is the RAG document metadata schema defined?",
        expected_sources=("schema.py",),
        category="rag_self",
    ),

    # ==============================================================
    # MULTILINGUAL
    # ==============================================================

    EvaluationCase(
        name="telugu_document",
        query="What Telugu test document is stored in the project?",
        expected_sources=("telugu_test.txt",),
        category="multilingual",
    ),

    EvaluationCase(
        name="human_language_detection",
        query="How does the RAG system detect human languages from text?",
        expected_sources=("detector.py",),
        category="multilingual",
    ),

    # ==============================================================
    # SIMILAR / COLLISION CASES
    # ==============================================================

    EvaluationCase(
        name="bone_driver_vs_editor",
        query="BoneDriver runtime implementation",
        expected_sources=("BoneDriver.cs",),
        expected_symbols=("BoneDriver",),
        software="Unity",
        project="Facial_Test",
        category="collision",
    ),

    EvaluationCase(
        name="lip_sync_window_vs_panel",
        query="FacialLipSyncWindow",
        expected_sources=("FacialLipSyncWindow.cs",),
        software="Unity",
        project="Facial_Test",
        category="collision",
    ),

    EvaluationCase(
        name="facial_automation_architecture_vs_capsule",
        query="facial automation architecture plan",
        expected_sources=(
            "MASTER_ARCHITECTURE_PLAN.md",
            "STRUCTURE_PLAN.md",
        ),
        software="Unity",
        project="Facial_Test",
        category="collision",
    ),

    # ==============================================================
    # UNKNOWN / OUT-OF-DOMAIN
    # ==============================================================

    EvaluationCase(
        name="unknown_customer_payments",
        query="What database schema does this project use for customer payments?",
        expected_sources=(),
        category="unknown",
    ),

    EvaluationCase(
        name="unknown_medical_system",
        query="How does this project process hospital patient medical records?",
        expected_sources=(),
        category="unknown",
    ),

    EvaluationCase(
        name="unknown_weather_system",
        query="What weather forecasting API does this project use?",
        expected_sources=(),
        category="unknown",
    ),

    EvaluationCase(
        name="unknown_ecommerce",
        query="What shopping cart and checkout system does this project implement?",
        expected_sources=(),
        category="unknown",
    ),
)