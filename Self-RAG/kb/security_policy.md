# AtlasWorks Security Policy (v1.0)

## Authentication
MFA is required for:
- corporate email
- source code hosting
- cloud console access
- Aurora admin console access

## Password rules
Passwords must:
- be at least 16 characters
- include at least 1 letter and 1 number
- not reuse any of the last 12 passwords

## Device security
All company laptops must use full-disk encryption.

### Lost or stolen devices
Lost or stolen company devices must be reported to IT within 12 hours.
If the device is lost while traveling internationally, report within 6 hours if possible.

## Data classification
AtlasWorks uses: Public, Internal, Confidential, Restricted.

### Restricted data
Restricted data includes:
- government IDs (passport number, national ID number, driver’s license number)
- bank account details (account number, routing number, IBAN)
- authentication secrets (API keys, private keys, OAuth refresh tokens)
- medical information
- precise location history (continuous location traces)

Restricted data must not be stored in personal cloud accounts or personal email.
Do not paste Restricted data into public issue trackers.

## Vulnerability SLAs
- Sev0 vulnerability in an internet-facing system: patch within 24 hours
- Sev1 vulnerability: patch within 7 days
- Sev2 vulnerability: patch within 30 days
