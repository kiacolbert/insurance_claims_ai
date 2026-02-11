# create_sample_pdf.py
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_sample_policy():
    c = canvas.Canvas("sample_policy.pdf", pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 100, "AUTO INSURANCE POLICY")
    
    # Content
    c.setFont("Helvetica", 12)
    y = height - 150
    
    sections = [
        ("Coverage Details", [
            "Liability: Up to $100,000 per person, $300,000 per accident",
            "Collision: $500 deductible",
            "Comprehensive: $250 deductible",
            "Uninsured motorist: Included"
        ]),
        ("Exclusions", [
            "Intentional damage",
            "Racing or speed contests",
            "Commercial use of vehicle",
            "Driving under the influence"
        ]),
        ("Claims Process", [
            "Report accident within 24 hours",
            "File claim online at www.insurance.com/claims or call 1-800-CLAIMS",
            "Provide police report if applicable",
            "Adjuster will contact you within 48 hours",
            "Repairs can begin after approval"
        ])
    ]
    
    for title, items in sections:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(100, y, title)
        y -= 25
        
        c.setFont("Helvetica", 12)
        for item in items:
            c.drawString(120, y, f"• {item}")
            y -= 20
        y -= 10
    
    c.save()
    print("✅ Created sample_policy.pdf")

if __name__ == "__main__":
    create_sample_policy()