// Your work experience entries
export type Experience = {
  role: string;
  company: string;
  type: string;
  duration: string;
  location?: string;
  bullets: string[];
};

export const experiences: Experience[] = [
  {
    role: "GenAI Developer",
    company: "Vionix Biosciences",
    type: "Seasonal",
    duration: "Apr 2025 - Aug 2025",
    location: "India · Remote",
    bullets: [
      "Led development of Retrieval-Augmented Generation (RAG) pipelines for scientific Q&A and spectroscopy queries, integrating document ingestion, embedding generation, and Milvus vector database with automated indexing from cloud storage (S3 & Google Drive).",
      "Engineered scalable async APIs and NL2SQL pipelines that converted natural language queries into SQL, enabling seamless metadata/file retrieval, automated database indexing, and secure cloud file access.",
    ],
  },
  {
    role: "Research Assistant",
    company: "Jadavpur University",
    type: "Apprenticeship",
    duration: "Oct 2024 - Mar 2025",
    bullets: [
      "Led development of multimodal AI solutions for detecting hateful content on social media achieving SOTA results on existing benchmarks.",
      "Extended methodology to multi-state cases using a contrastive learning approach.",
    ],
  },
  {
    role: "Data Scientist",
    company: "SoftTechNation (Confidential Client)",
    type: "Contract",
    duration: "Jul 2024 - Feb 2025",
    location: "Remote",
    bullets: [
      "Contributed to 2 GenAI + RAG projects for confidential industry clients.",
    ],
  },
  {
    role: "Data Science Analyst",
    company: "Urbanlyfe",
    type: "Part-time",
    duration: "Nov 2023 - Mar 2024",
    location: "Gujarat, India · Remote",
    bullets: [
      "Worked on prototyping a chatbot for solving user queries and product recommendations using Botpress for UrbanLyfe App. Experimented with fine-tuning LLM models for enhancing response.",
      "Explored and brainstormed the utilization of ML methods and BI dashboards for the logistics tracking and cost optimization products.",
    ],
  },
  {
    role: "NLP Intern",
    company: "Defence Research & Development Organisation (DRDO)",
    type: "Apprenticeship",
    duration: "Jun 2023 - Aug 2023",
    location: "New Delhi, India · Hybrid",
    bullets: ["Worked on Speech Processing Models."],
  },
  {
    role: "Undergraduate Research Assistant",
    company: "Atma Ram Sanatan Dharma College",
    type: "Part-time",
    duration: "Sep 2022 - Apr 2023",
    location: "South Delhi, India · On-site",
    bullets: [
      "Balanced code-mixed text data for unbiased sarcasm classification using four text augmentation and three data balancing techniques.",
      "Fine-tuned EfficientNetB7 on Alzheimer's MRI scans with transfer learning, achieving improved accuracy for early diagnosis.",
      "Developed a text detection and localization system using Transformers.",
    ],
  },
  {
    role: "Research Intern",
    company: "National Institute of Technology Srinagar",
    type: "Internship",
    duration: "Jun 2022 - Mar 2023",
    location: "Srinagar, India · Remote",
    bullets: [
      "Evaluated scalability, throughput, and latency of Solidity-based blockchain systems using benchmarking and profiling techniques.",
      "Currently writing a book on fundamental principles of blockchain technology.",
      "Researched applicability of Blockchain in Intrusion Detection Systems for CPS, UAVs and IoT systems.",
    ],
  },
  {
    role: "Research Intern",
    company: "National University of Malaysia (UKM)",
    type: "Internship",
    duration: "Sep 2022 - Dec 2022",
    location: "Bangi, Malaysia · Remote",
    bullets: [
      "Researched 'enabled transfer learning for low-resource languages'. Fine-tuned BERTs for domain-specific tasks.",
      "Conducted statistical text analytics and data augmentation for various NLP tasks including hate speech detection.",
      "Contributed to spear-phishing analysis on UKM students using simulated emails mimicking real attacks.",
    ],
  },
];

export type Education = {
  degree: string;
  institution: string;
  duration: string;
  grade?: string;
  details?: string[];
};

export const education: Education[] = [
  {
    degree: "Masters in Computer Vision (Erasmus Mundus IPCVai)",
    institution: "Universidad Autónoma de Madrid",
    duration: "Feb 2026 – Jun 2026",
  },
  {
    degree: "Masters in Computer Vision (Erasmus Mundus IPCVai)",
    institution: "Pázmány Péter Katolikus Egyetem",
    duration: "Sep 2025 – Jan 2026",
    grade: "4.8/5",
    details: [
      "Basic Image Processing Algorithms (Core)",
      "Biomedical Signal Processing (Elective)",
      "Data Mining and Machine Learning (Core)",
      "Fundamentals and Basic Tools for Deep Learning (Core)",
      "Design Patterns (Elective)",
      "Multimodal Sensor Fusion and Navigation (Core)",
    ],
  },
  {
    degree: "B.Sc. (H) Computer Science",
    institution: "Atma Ram Sanatan Dharma College, Delhi University",
    duration: "2020 – 2023",
  },
];
