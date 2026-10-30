export type Experience = {
  role: string;
  company: string;
  companyLink?: string;
  type: string;
  duration: string;
  location?: string;
  supervisor?: string;
  supervisorLink?: string;
  bullets: string[];
};

export const experiences: Experience[] = [
  {
    role: 'Computer Vision Intern', company: 'Vaelsys', companyLink: 'https://vaelsys.com/', type: 'Internship', duration: 'Jun 2026 – Aug 2026', location: 'Madrid, Spain', supervisor: 'Dr. Eduardo Cermeño', supervisorLink: 'https://www.linkedin.com/in/eduardo-cerme%C3%B1o-phd-7992293/',
    bullets: [
      'Built an edge video threat-detection pipeline for the Vaelsys–Intel proof of concept, benchmarking 5+ VLMs and ML classifiers with up to 95% theft recall.',
      'Benchmarked Qwen3-VL-8B inference on Intel iGPU through OpenVINO Model Server and gRPC, supporting 8–12 concurrent camera streams.',
      'Stress-tested an NPU–GPU two-stage edge-AI architecture combining D-FINE, DLStreamer, GStreamer, VLM inference, and ML classification.',
      'Led a VLM reliability study on counterfactual anomaly-region grounding for industrial inspection across 13k+ evaluations.',
    ],
  },
  {
    role: 'GenAI Engineer', company: 'Vionix Biosciences', companyLink: 'https://www.vionix.bio/', type: 'Contract', duration: 'Apr 2025 – Aug 2025', location: 'USA · Remote', supervisor: 'Dr. Jagadish Venkataraman', supervisorLink: 'https://www.linkedin.com/in/jagadish-venkataraman-83b4074/',
    bullets: ['Led RAG pipelines for spectroscopy Q&A, integrating embeddings, Milvus, and automated cloud indexing.', 'Built asynchronous APIs and NL2SQL pipelines for automated metadata and file retrieval with secure cloud access.', 'Recognized by executive leadership for high-impact delivery and offered a permanent role.'],
  },
  {
    role: 'Data Scientist', company: 'SoftTechNation', companyLink: 'https://www.linkedin.com/company/softtech-nation/', type: 'Part-time', duration: 'Sep 2024 – Dec 2024', location: 'Remote',
    bullets: ['Developed a RAG web application with AstraDB and Groq using Llama embeddings for efficient loading, chunking, and storage.', 'Worked across MLOps and ETL with Docker, PySpark, Airflow, and dbt.'],
  },
  {
    role: 'Analyst', company: 'UrbanLyfe', companyLink: 'https://www.linkedin.com/company/urban-lyfe/', type: 'Part-time', duration: 'Nov 2023 – Mar 2024', location: 'Gujarat, India · Remote',
    bullets: ['Built a Botpress chatbot for automated product queries and explored LLM fine-tuning for improved responses.', 'Explored applied ML and BI techniques to integrate retail stores and optimize logistics sustainably.'],
  },
  {
    role: 'Research Intern', company: 'University College Dublin & University of Cincinnati', type: 'Research', duration: 'Jul 2023 – Mar 2024', location: 'Remote', supervisor: 'Dr. Soumyabrata Dev', supervisorLink: 'https://people.ucd.ie/soumyabrata.dev',
    bullets: ['Developed a fusion encoder–decoder model for emotion-enriched image captions with pretrained deep-learning features.', 'Fine-tuned BERT and RoBERTa for adversarial prompt detection to prevent LLM jailbreaking.', 'Performed sentiment analysis on TikTok reviews to infer psychological behaviour.'],
  },
  {
    role: 'Project Trainee', company: 'Defence Research and Development Organisation', companyLink: 'https://drdo.gov.in/drdo/en', type: 'Traineeship', duration: 'Jun 2023 – Aug 2023', location: 'New Delhi, India', supervisor: 'Sci. Gaurav Gupta',
    bullets: ['Developed an offline speech-detection system with OpenAI Whisper, reaching 91% accuracy and 4.2% WER.', 'Implemented speech-recognition to audio-segmentation conversion followed by WAV conversion for uninterrupted audio input.'],
  },
];

export type Education = {
  degree: string;
  institution: string;
  institutionLink?: string;
  duration: string;
  location?: string;
  grade?: string;
  details?: string[];
};

export const education: Education[] = [
  { degree: 'Erasmus Mundus – Image Processing & Computer Vision (IPCVai)', institution: 'Pázmány Péter Catholic University', institutionLink: 'https://ppke.hu/', duration: 'Sep 2025 – Jun 2027 (expected)', location: 'Budapest, Hungary', details: ['EU Full Scholarship · selected among 19 awardees from 1,700+ global applicants.'] },
  { degree: 'Erasmus Mundus – Image Processing & Computer Vision (IPCVai)', institution: 'Universidad Autónoma de Madrid', institutionLink: 'https://www.uam.es/', duration: 'Sep 2025 – Jun 2027 (expected)', location: 'Madrid, Spain' },
  { degree: 'Erasmus Mundus – Image Processing & Computer Vision (IPCVai)', institution: 'Université de Bordeaux', institutionLink: 'https://www.u-bordeaux.com/', duration: 'Ongoing', location: 'Bordeaux, France' },
  { degree: 'B.Sc. (Hons.) Computer Science · Minor in Mathematics', institution: 'University of Delhi – ARSD College', institutionLink: 'https://arsdcollege.ac.in/', duration: 'Jul 2020 – Jun 2023', location: 'New Delhi, India', details: ['University of Delhi: https://www.du.ac.in/'] },
];

export const skills = {
  Languages: ['Python', 'C/C++', 'JavaScript', 'Java', 'SQL', 'PostgreSQL', 'Solidity'],
  Frameworks: ['TensorFlow', 'PyTorch', 'OpenCV', 'HuggingFace', 'Scikit-learn', 'DVC', 'AWS'],
  Tools: ['VS Code', 'Power BI', 'Git', 'Spyder', 'RStudio', 'LaTeX', 'Excel'],
};

export const achievements = [
  'Awarded 8+ fully and partially funded graduate admissions offers, including EMJMD and MBZUAI offers.',
  'Secured a spot in the Innomatics Research Lab Traineeship among the top 5% of 20,000+ applicants.',
  'Selected from Delhi University for the ISSA, DRDO Traineeship.',
  'Won a research-writing event at Kerala Blockchain Academy.',
  'Placed among the top 15 of approximately 380 participants at Its Data Time, organized by IIITNR.',
  'Ranked second in a Mathematical Writing Competition at Gargi College, University of Delhi.',
];
