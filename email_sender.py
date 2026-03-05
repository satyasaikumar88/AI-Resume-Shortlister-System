import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmailSender:
    """
    Sends emails via Office 365 or Gmail.
    Uses STARTTLS/587 for compatibility.
    """

    def __init__(self, smtp_server, smtp_port, sender_email, sender_password):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password

    def _build_message(self, recipient_email, name, role, match_percentage):
        """Construct the MIME email message."""
        subject = f"Congratulations {name}! You have been shortlisted for the {role} role"
        body = (
            f"Dear {name},\n\n"
            f"We are pleased to inform you that your resume has been shortlisted "
            f"for the {role} role with a match score of {match_percentage:.2f}%.\n\n"
            f"Our recruitment team will reach out to you shortly with the next steps "
            f"in the selection process.\n\n"
            f"Best regards,\n"
            f"HR Team\n"
            f"Woxsen University"
        )
        msg = MIMEMultipart()
        msg["From"] = self.sender_email
        msg["To"] = recipient_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))
        return msg

    def send_email(self, recipient_email, name, role, match_percentage):
        """
        Send a shortlist notification email using SMTP.
        Returns True on success, False on failure.
        """
        if not recipient_email or str(recipient_email).strip() in ("", "N/A"):
            logger.warning("No valid recipient email — skipping.")
            return False

        recipient_email = str(recipient_email).strip()
        logger.info(f"Sending email to: {recipient_email} (role={role}, score={match_percentage:.2f}%)")
        logger.info(f"SMTP Server: {self.smtp_server}:{self.smtp_port}")

        msg = self._build_message(recipient_email, name, role, match_percentage)

        server = None
        try:
            # Connect to SMTP server
            logger.info(f"Connecting to {self.smtp_server}:{self.smtp_port}...")
            server = smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=30)
            server.set_debuglevel(1)  # Enable debug output
            
            # Start TLS encryption
            logger.info("Starting TLS...")
            server.starttls()
            
            # Login
            logger.info(f"Logging in as {self.sender_email}...")
            server.login(self.sender_email, self.sender_password)
            
            # Send email
            logger.info(f"Sending email to {recipient_email}...")
            server.send_message(msg)
            
            logger.info(f"✅ Email successfully sent to {recipient_email}")
            return True

        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"❌ Authentication failed: {e}")
            logger.error("Check your email and password in .env file")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"❌ SMTP error: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Error sending email: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        finally:
            if server:
                try:
                    server.quit()
                except Exception:
                    pass