import streamlit as st
import base64

def inject_custom_css():
    with open("assets/styles.css") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def styled_header(text):
    st.markdown(f"""
    <div class="custom-card">
        <h2 style="color: #1e88e5; margin: 0;">{text}</h2>
    </div>
    """, unsafe_allow_html=True)

def display_total(amount):
    st.markdown(f"""
    <div class="custom-card total-display">
        <h3 style="margin: 0;">Total Amount</h3>
        <h1 style="margin: 0;">Rs.{amount}</h1>
    </div>
    """, unsafe_allow_html=True)

def display_bill(df):
    if df.empty:
        st.info("No items detected yet")
        return
    
    # Convert dataframe to HTML with styling
    html = """
    <div class="custom-card">
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr style="border-bottom: 2px solid #1e88e5;">
                    <th style="text-align: left; padding: 8px;">Item</th>
                    <th style="text-align: center; padding: 8px;">Qty</th>
                    <th style="text-align: right; padding: 8px;">Total</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for _, row in df.iterrows():
        html += f"""
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 8px;">{row['Item']}</td>
                <td style="text-align: center; padding: 8px;">{row['Quantity']}</td>
                <td style="text-align: right; padding: 8px;">Rs.{row['Total']}</td>
            </tr>
        """
    
    html += """
            </tbody>
        </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)