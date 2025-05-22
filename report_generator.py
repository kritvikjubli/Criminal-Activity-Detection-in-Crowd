import os
from reportlab.pdfgen import canvas
import sqlite3
import datetime
import matplotlib.pyplot as plt

def generate_report():
    if not os.path.exists("reports"):
        os.makedirs("reports") 

    conn = sqlite3.connect("reports.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM detections ORDER BY timestamp DESC LIMIT 10")
    detections = cursor.fetchall()
    conn.close()

    if not detections:
        return None

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_filename = f"reports/report_{timestamp}.pdf"

    # Generating charts
    labels = [det[1] for det in detections]
    confidences = [det[2] for det in detections]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, confidences)
    bar_chart_filename = f"reports/bar_chart_{timestamp}.png"
    plt.savefig(bar_chart_filename)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.pie(confidences, labels=labels, autopct='%1.1f%%')
    pie_chart_filename = f"reports/pie_chart_{timestamp}.png"
    plt.savefig(pie_chart_filename)
    plt.close()

    # Create PDF report
    c = canvas.Canvas(report_filename)
    c.drawString(100, 750, "Suspicious Activity Report")
    y = 720
    for det in detections:
        c.drawString(100, y, f"{det[1]} - Confidence: {det[2]:.2f} - {det[3]}")
        y -= 20

    c.drawImage(bar_chart_filename, 100, y - 200, width=400, height=200)
    c.drawImage(pie_chart_filename, 100, y - 400, width=400, height=200)

    c.save()
    return report_filename