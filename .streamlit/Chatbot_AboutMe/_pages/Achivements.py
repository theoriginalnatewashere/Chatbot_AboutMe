import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
st.title("Achivements")
Rage_badge = """
<div data-iframe-width="150" data-iframe-height="270" data-share-badge-id="365e7215-70ac-4ce6-8556-45611fb09f47" data-share-badge-host="https://www.credly.com"></div>
<script type="text/javascript" async src="//cdn.credly.com/assets/utilities/embed.js"></script>
"""
Hope_Badge = """<div data-iframe-width="150" data-iframe-height="270" data-share-badge-id="e3b54dd6-1ad7-462d-95ee-75491fb55c2f" data-share-badge-host="https://www.credly.com"></div><script type="text/javascript" async src="//cdn.credly.com/assets/utilities/embed.js"></script>"""
Keyholders_badge = """<div data-iframe-width="150" data-iframe-height="270" data-share-badge-id="0385f6ff-609e-4ed1-9234-fce7dd813a88" data-share-badge-host="https://www.credly.com"></div><script type="text/javascript" async src="//cdn.credly.com/assets/utilities/embed.js"></script>"""
Azure_fundamentals = """<div data-iframe-width="150" data-iframe-height="270" data-share-badge-id="667db6e7-111e-4a71-9028-cfcbb512cdcf" data-share-badge-host="https://www.credly.com"></div><script type="text/javascript" async src="//cdn.credly.com/assets/utilities/embed.js"></script>"""
# Hackathons section
st.header("💻 Hackathons")
st.write("**Dell Hackathon Hack to Hire 2022** - Winner")
st.write("**Cybersecurity Hackathon 2021** - Winner")

# Capture-the-flag section
st.header("🔓 Capture-the-Flag Competitions")
st.write("**Capture the Flag Hope 2022** - Solved")
st.write("**Capture the Flag Tempus 2022** - Keyholder")
st.write("**NoShitSecurity Jurassic Jungle Season 2** - Graduate")

# Awards section
st.header("🏅 Awards")
st.write("**National Service Scheme Best Volunteer 2022**")
st.write("**Socity For Promotion of Youth And Masses Best Volunteer 2021**")
# Certifications section
st.header("📜 Certifications")

# Creating a dataframe for certifications
certifications_data = {
    "Vendor": ["Microsoft", "Microsoft", "NoShitSecurity", "Great Learning", "Splunk", "AttackIQ", "Coursera", "Try Hack Me", "Cybrary", "Google"],
    "Series": ["AZ-500 Azure Security Engineer", "AZ-900 Azure Fundamentals", "Azure Security Bootcamp", "Cloud Fundamentals", "Basic of Splunk", "Foundational and operationalizing Mitre Att&ack", "HTML, CSS, and JavaScript for Web Developers", "Advent of Cyber 3", "Introduction to IT and Cybersecurity", "Google Cloud Engineering Track"],
    "Awarded": ["Aug/2023", "Oct/2022", "July/2022", "June/2022", "May/2022", "May/2022", "Feb/2022", "Dec/2021", "Nov/2021", "Oct/2021"],
    "Expiration": ["Aug/2024", "Never", "Never", "Never", "Never", "Never", "Never", "Never", "Never", "Never"]
}

certifications_df = pd.DataFrame(certifications_data)
st.table(certifications_df)
#Badge Section
st.header("Badges")
col1, col2, col3, col4 = st.columns(4)

with col1:
    components.html(Rage_badge, height=270, width=165)
with col2:
    components.html(Hope_Badge, height=270, width=165)
with col3:
    components.html(Keyholders_badge, height=270, width=165)
with col4:
    components.html(Azure_fundamentals, height=270, width=165)
