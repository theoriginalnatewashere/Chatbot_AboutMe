# Define the left and right columns
right_column, left_column = st.columns([1, 2])
with right_column:
    st.markdown("<br>", unsafe_allow_html=True)
    st.image(r"images/profile.png", use_column_width=True)

with left_column:
   st.title("About Me")
st.markdown(
    "<div style='text-align: justify'>"
    "Welcome. My name is Nethan. I have a background in industrial design and currently specialize in data-driven design, working at the intersection of design, data science, and emerging technologies. My work focuses on creating responsible, accessible, and human-centered AI systems."
    "<br><br>"
    "With experience across entrepreneurship, applied research, and digital innovation, I contribute to projects involving ethical AI, healthcare system optimization, stakeholder mapping, and data ethics. I have collaborated with organizations such as Datakluis and participated in initiatives promoting community-driven innovation and digital privacy."
    "<br><br>"
    "My academic background includes a Master of Data-Driven Design, supported by certifications in data science, human-centered design, and privacy-by-design frameworks. I am especially interested in human-AI collaboration, emergency response systems, and improving societal well-being through ethical technology."
    "<br><br>"
    "Thank you for visiting my portfolio. Please feel free to connect through the links below."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div style='text-align: justify'>"
    "<br>"
    "<a href='https://linkedin.com/in/natewashere'><img src='https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png' width='40'></a>"
    "&nbsp;&nbsp;&nbsp;&nbsp;"
    "<a href='https://github.com/your-github-handle'><img src='https://upload.wikimedia.org/wikipedia/commons/2/24/Github_logo_svg.svg' width='40'></a>"
    "&nbsp;&nbsp;&nbsp;&nbsp;"
    "<a href='mailto:nate@digitalnomad.jp'><img src='https://upload.wikimedia.org/wikipedia/commons/4/4e/Gmail_Icon.png' width='40'></a>"
    "</div>",
    unsafe_allow_html=True,
)

