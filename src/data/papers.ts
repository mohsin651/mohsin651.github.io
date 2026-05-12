// All your research papers - just add new entries to add papers
export type Paper = {
  title: string;
  authors: string;
  venue: string;
  year: number;
  status?: "Published" | "Under Review" | "Accepted" | "In Progress";
  link?: string;
  pdf?: string;
  abstract?: string;
  selected?: boolean; // show on homepage
};

export const papers: Paper[] = [
  {
    title: "DAMM: Dynamic Modality Aware Weighted Embeddings Fusion for Multimodal Meme Detection",
    authors: "Mohsin Imam, Utathya Aich, Ram Sarkar",
    venue: "Information Fusion, Elsevier",
    year: 2024,
    status: "Under Review",
    selected: true,
  },
  {
    title: "On Utilizing Deep Learning Models for Preventing Harmful Response in LLMs",
    authors: "Mohsin Imam, Salehah Hamzah, Shalini Aggarwal, Soumyabrata Dev, V.B Surya Prasath",
    venue: "Neural Networks, Elsevier",
    year: 2024,
    status: "Under Review",
    pdf: "/papers/LLM-Paper.pdf",
    selected: true,
    abstract: "Past few years have witnessed tremendous progress in artificial intelligence (AI) technologies including but not limited to natural language processing (NLP), metaverse, and generative models. To enhance the security of LLMs and prevent them from responding to adversarial prompts, we utilize machine learning, deep learning and transformer models as external prompt classifiers. Our proposed ensemble of the BERT-DistilBERT model achieved 97.56% accuracy in identifying malicious/adversarial prompts.",
  },
  {
    title: "Air Quality Monitoring Using Statistical Learning Models for Sustainable Environment",
    authors: "Mohsin Imam, Sufiyan Adam, Soumyabrata Dev, Nashreen Nesa",
    venue: "Intelligent Systems with Applications, Elsevier",
    year: 2024,
    status: "Published",
    link: "https://www.sciencedirect.com/science/article/pii/S2667305324000097",
    selected: true,
  },
  {
    title: "An Ensemble Method for Categorizing Cardiovascular Disease",
    authors: "Mohsin Imam, Sufiyan Adam, Neetu Agarwal, Suyash Kumar, Anjana Gosain",
    venue: "International Conference on Advances in IoT and Security with AI",
    year: 2023,
    status: "Published",
    link: "https://link.springer.com/chapter/10.1007/978-981-99-5088-1_24",
  },
  {
    title: "Study of Student Personality Trait on Spear-Phishing Susceptibility Behaviour",
    authors: "Mohamad Alhaddad, Masnizah Mohd, Faizan Qamar, Mohsin Imam",
    venue: "International Journal of Advanced Computer Science and Applications (IJACSA), 14(5)",
    year: 2023,
    status: "Published",
    link: "https://thesai.org/Publications/ViewPaper?Volume=14&Issue=5&Code=IJACSA&SerialNo=71",
  },
  {
    title: "Review on Applications and Security of Blockchain",
    authors: "Mohsin Imam, Kavita Saini",
    venue: "Book Chapter: Blockchain and EHR, Nova Science Publisher",
    year: 2023,
    status: "Published",
    link: "https://novapublishers.com/shop/blockchain-and-ehr/",
  },
  {
    title: "Cyber Threat Analysis and Mitigation in Emerging Information Technology (IT) Trends",
    authors: "Mohsin Imam, Mohd Anas Wajid, Bharat Bhushan, Alaa Ali Hameed, Akhtar Jamil",
    venue: "International Conference on Emerging Trends and Applications in Artificial Intelligence",
    year: 2023,
    status: "Published",
    link: "https://www.springerprofessional.de/en/cyber-threat-analysis-and-mitigation-in-emerging-information-tec/27041128",
  },
  {
    title: "Comparative Analysis of Deep Learning Models for Alzheimer's Disease Stage Classification Using Transfer Learning",
    authors: "Archana Gahlaut, Mohsin Imam",
    venue: "IEEE ICCCIS",
    year: 2023,
    status: "Published",
    link: "https://ieeexplore.ieee.org/document/10425684",
  },
  {
    title: "Precision Location Keyword Detection Using Offline Speech Recognition Technique",
    authors: "Mohsin Imam, Gaurav Gupta",
    venue: "Preprints.org",
    year: 2023,
    status: "Published",
    link: "https://www.preprints.org/manuscript/202310.0690/v2",
  },
  {
    title: "Parametric Optimization and Comparative Study of Machine Learning and Deep Learning Algorithms for Breast Cancer Diagnosis",
    authors: "Parul Jain, Shalini Gupta, Sufiyan Adam, Mohsin Imam",
    venue: "Breast Disease, IOS Press",
    year: 2024,
    status: "Published",
    link: "https://content.iospress.com/articles/breast-disease/bd240018",
  },
  {
    title: "Assessing AI Chatbot Responses in Promoting COVID-19 Vaccine Acceptance",
    authors: "Mohsin Imam, Abdul Quershi",
    venue: "Social Science Research Network",
    year: 2024,
    status: "Published",
    link: "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4851422",
  },
  {
    title: "Adversarial Learning of Security Risk in Cobots Driven Industry",
    authors: "Mohsin Imam, Mohd Anas Wajid, Aasim Zafar, Mohammad Saif Wajid, Neeraj Baishwar, Sameer Awasthi",
    venue: "Book chapter: Intersection of Machine Learning and Computational Social Sciences, CRC Press",
    year: 2024,
    status: "Accepted",
  },
  {
    title: "Applications of Multimodal Deep Learning in Medicine",
    authors: "Pankaj Rajdeo, ..., Mohsin Imam, VB Surya Prasath",
    venue: "Springer Handbook on Medical Biotechnology",
    year: 2024,
    status: "Under Review",
  },
  {
    title: "Enhancing BERT-Based Models for Detecting Race, Violence, and Hate Speech in Low-Resource Settings",
    authors: "Salehah Hamzah, Mohsin Imam, Masnizah Mohammad",
    venue: "Journal Name",
    year: 2024,
    status: "Under Review",
  },
  {
    title: "Building Blockchain-powered Decentralized Applications with Solidity: A Project-Oriented Guide",
    authors: "Sparsh Sharma, Mohsin Imam",
    venue: "To be Announced",
    year: 2024,
    status: "In Progress",
  },
];
