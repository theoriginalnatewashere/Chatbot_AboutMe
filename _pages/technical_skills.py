from streamlit_pills import pills
import json
# Function to load a JSON file

with open(r'data/technical_skills.json') as file:
    description = json.load(file)

right_column, left_column = st.columns([2, 1])
with right_column:
    st.title("Techincal Skills")
    skills = [
        "Microservices",
        "Large Language Model (LLM)",
        "Small Language Model (SLM)",
        "Fine Tuning",
        "Vector Database",
        "Prompt Engineering",
        "Docker",
        "Kubernetes",
        "RabbitMQ",
        "Orchestration",
        "Golang",
        "MongoDB",
        "API",
        "Data Mining",
        "Data Analytics",
        "JSON",
        "Python/R",
        "Data Visualization",
        "Orange Framework",
        "Jupyter Notebook",
        "Excel",
        "Machine Learning Algorithms",
        "Automation",
        "ETL Pipeline",
        "Azure Cloud Architecture",
        "Azure Security Engineering",
        "Azure Monitor",
        "Virtual Machine",
        "Virtual Networks",
        "Cosmos DB",
        "Disaster Recovery",
        "Azure Firewall and Defender",
        "Azure Functions",
        "Azure Blueprints",
        "Open-source Intelligence (OSINT)",
        "Splunk",
        "Microsoft Azure Active Directory",
        "MITRE ATT&CK",
        "Active Directory",
        "Architecture",
        "Auditing",
        "Azure",
        "Bash",
        "Blue Team",
        "Cisco",
        "Cloud",
        "Cloud Systems & Endpoints Active Directory Management",
        "Cryptography and Encryption Tools",
        "Network and System Scanning Tools",
        "Code",
        "Compliance",
        "Containers",
        "Cryptosystems",
        "Defender",
        "Encryption",
        "GitLab",
        "Linux",
        "Lynis",
        "Metasploit",
        "NIST",
        "Policy",
        "PowerShell",
        "Python",
        "Red Team",
        "Reporting",
        "Sentinel",
        "SIEM",
        "Splunk",
        "Strategy",
        "Technical Writing",
        "Unix",
        "VPN",
        "Vulnerability Assessment",
        "Window Server",
    ]
    selected = pills("Select a category", skills)
with left_column:
    st.title("Description")
    for skill in description["skills"]:
        if selected==skill["name"]:
            st.markdown(skill["description"])