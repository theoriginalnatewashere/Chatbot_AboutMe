import reveal_slides as rs

sample_markdown = r"""
# IINDUSTRY EXPERIENCE
Brief overview of my experience till date.
---


## Dell Technologies
`Software Engineer Winter Intern`
</br>
``Jan-May 2024``
<div style='text-align: justify'><b>
<li>
Developed and implemented features for a state-based API orchestration tool, ensuring seamless integration and coordination of various services within a microservices-based architecture.</li><br>
<li>Designed chatbots and tools to provide insights and information regarding orders using Prompt Engineering.</li><br> 
<li>Conducted research and developed a comprehensive report on handling NULL scenarios in MongoDB, implementing solutions for efficient and error-free reading and writing operations.</li><br>
</b>
</div>
Technologies Used
--
<b>Technologies Used</b><br>
<!-- .slide: data-background-color="#283747" -->
<div style='text-align: justify'>
Microservices, Large Language Model (LLM), Small Language Model (SLM),Fine Tuning, Vector Database, Prompt Engineering,Docker, Kubernetes, RabbitMQ, orchestration, Golang, Python, MongoDB, API
</div>
---
## NoShitSecurity
`Cloud Architecture and Engineering Intern`
</br>
`Dec-Aug 2023`
<div style='text-align: justify'><b>
<li>Develops and deploys resilient Azure cloud infrastructure.</li><br>
<li>Evaluates security posture, pinpoints strengths and weaknesses, applies forward-thinking, and recommends remediations</li><br> 
<li>Utilizes the latest industry-leading trends and techniques to host a worldwide capture-the-flag event.</li><br> 
<li>Develops or implements open-source or third-party tools to reduce exposure or secure cloud infrastructure</li><br>
</b>
</div>
Technologies Used
--
<b>Technologies Used</b><br>
<!-- .slide: data-background-color="#283747" -->
<div style='text-align: justify'>
Azure Cloud Architecture, Azure Security Engineering,Azure Monitor, virtual machine, virtual networks, Cosmos DB, Disaster Recovery, Azure Firewall and Defender, Azure Functions, Azure Blueprints, Cloud Systems & Endpoints Active Directory Management, Cryptography and encryption tools, Network and system scanning tools, Open-source intelligence (OSINT)

</div>
---
## Dell Technology
`Software Engineer Summer Intern`
</br>
`May-July 2023`
<div style='text-align: justify'><b>
<li>Collaborated with the team to implement various automation tools, ensuring proper flow and efficient management of orders.</li><br>
<li>Prediction of order cancellation using machine learning algorithms. </li><br> 
<li>Examine the factors that contribute to order cancellation by identifying and addressing system gaps.</li><br> 
<li>Analyzed data to identify factors contributing to order cancellations and system holds, addressing system gaps to reduce order cancellations and improve overall efficiency.</li><br>
</b>
</div>
Technologies Used
--
<b>Technologies Used</b><br>
<!-- .slide: data-background-color="#283747" -->
<div style='text-align: justify'>
Data Mining, Data Analytics, JSON, Python/R, Data Visualization, Orange Framework, Jupyter Notebook, Excel, Machine learning algorithms, Automation, ETL pipeline.

</div>
---
## NoShitSecurity
`Intern`
</br>
`June 2022 -April 2023`
<div style='text-align: justify'><b>
<li>Designs and conducts research for securing cloud infrastructure.</li><br>
<li>Uses advanced configuration hardening techniques to improve the security of VMs and workloads.</li><br> 
<li>Assures system integrity with industry-standard policy and defense-in-depth security controls.</li><br> 
<li>Leverages Azure Defender tooling to produce and maintain a healthier security posture  </li><br>
</b>
</div>
Technologies Used
--
<b>Technologies Used</b><br>
<!-- .slide: data-background-color="#283747" -->
<div style='text-align: justify'>
Linux, Lynis, Metasploit, NIST, Policy, PowerShell, Python, Red Team, Reporting, Sentinel, SIEM, Splunk, Strategy, Technical Writing, Unix VPN, Vulnerability Assessment, Window Server

</div>
---
## Virtually Testing Foundation 
`Cyber Security Intern`
</br>
`May-July 2022`
<div style='text-align: justify'><b>
<li>Maintains technical knowledge by attending multiple workshops and online courses </li><br>
<li>Manage and produce technical documentation, process guides, and related content on cybersecurity</li><br> 
<li>Successfully completed all assigned capture-the-flag challenges</li><br> 
</b>
</div>
Technologies Used
--
<b>Technologies Used</b><br>
<!-- .slide: data-background-color="#283747" -->
<div style='text-align: justify'>
Basic cloud security and engineering, Splunk, Open-source Intelligence (OSINT), Microsoft Azure Active Directory, MITRE ATT&CK 
</div>
"""
st.title("Experience")
currState = rs.slides(
    sample_markdown,
    theme="dracula",
    height=600,
    config={
        "transition": "slide",
        "width": 1100,
        "height": 1100,
        "center": True,
        "margin": 0.10,
        "scale_range": [0.1, 3.0],
    },
    initial_state={
        "indexf": -1,
    },
    markdown_props={"data-separator-vertical": "^--$"},
    key="foo",
)
