import streamlit as st


def icon(icon_names, links):
    st.markdown(f'<a href="{links[0]}" title="Github Repository"><i class="{icon_names[0]}"></i></a>'
                f' <a href="{links[1]}" title="Web Application"><i class="{icon_names[1]}"></i></a>',
                unsafe_allow_html=True)