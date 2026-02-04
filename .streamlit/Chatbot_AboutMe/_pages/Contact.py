st.header(":mailbox: Get In Touch With Me!")


contact_form = """
<form action="https://formsubmit.co/86806d01587674667580d00061ad9f49" method="POST">
     <input type="hidden" name="_captcha" value="true">
     <input type="text" name="name" placeholder="Your name" required>
     <input type="email" name="email" placeholder="Your email" required>
     <textarea name="message" placeholder="Your message here"></textarea>
     <button type="submit">Send</button>
</form>
"""

st.markdown(contact_form, unsafe_allow_html=True)
# Use Local CSS File
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css(r"style/style.css")