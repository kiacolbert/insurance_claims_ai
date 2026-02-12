# create_sample_policies.py
"""
Add sample insurance policy documents to test with
"""

import json

# Create realistic policy documents
sample_policies = [
    {
        "question": "Auto Insurance Policy Overview",
        "content": """AUTO INSURANCE POLICY - SUMMARY

POLICYHOLDER INFORMATION:
Policy Number: AUT-2024-12345
Effective Date: January 1, 2024
Coverage Period: 12 months

COVERAGE A: LIABILITY
- Bodily Injury: $100,000 per person / $300,000 per accident
- Property Damage: $50,000 per accident

COVERAGE B: COLLISION
- Deductible: $500 per accident
- Covers damage to your vehicle in a collision
- Includes hit-and-run accidents

COVERAGE C: COMPREHENSIVE
- Deductible: $250 per incident
- Covers theft, vandalism, weather damage, animal collisions
- Glass damage: No deductible

COVERAGE D: UNINSURED MOTORIST
- Bodily Injury: $100,000 per person / $300,000 per accident
- Property Damage: $50,000 per accident

EXCLUSIONS:
- Intentional damage or destruction
- Racing or speed contests
- Commercial use of personal vehicle
- Driving under the influence
- Normal wear and tear

PREMIUM: $1,245 annually ($103.75/month)"""
    },
    {
        "question": "Claims Filing Process",
        "content": """HOW TO FILE AN INSURANCE CLAIM

STEP 1: IMMEDIATE ACTIONS (Within 24 hours)
- Ensure safety of all parties
- Call 911 if there are injuries or significant damage
- Do NOT admit fault at the scene
- Exchange information with other driver(s):
  * Name, phone number, address
  * Insurance company and policy number
  * Driver's license number
  * License plate number
  * Vehicle make, model, year

STEP 2: DOCUMENT THE INCIDENT
- Take photos of:
  * All vehicle damage (multiple angles)
  * Accident scene (road conditions, traffic signs)
  * Other vehicles involved
  * Any visible injuries
- Get contact info from witnesses
- Obtain police report number if applicable

STEP 3: REPORT THE CLAIM
Contact us within 24 hours:
- Phone: 1-800-555-CLAIM (available 24/7)
- Online: www.insurance-company.com/file-claim
- Mobile App: Tap "File New Claim"

Information needed:
- Your policy number
- Date, time, and location of incident
- Description of what happened
- Police report number (if applicable)
- Photos of damage
- Other driver's insurance information

STEP 4: ADJUSTER ASSIGNMENT
- Claim adjuster assigned within 24-48 hours
- Adjuster will contact you to schedule vehicle inspection
- Inspection typically occurs within 3-5 business days

STEP 5: REPAIR AUTHORIZATION
- Repair estimate prepared within 24 hours of inspection
- You'll receive approval to proceed with repairs
- Choose from our network of approved repair shops or your own
- Rental car coverage (if included) begins once repairs approved

TYPICAL TIMELINE:
- Initial contact: Immediate
- Adjuster assignment: 24-48 hours
- Vehicle inspection: 3-5 business days
- Repair approval: 1-2 business days after inspection
- Total process: 7-14 days from filing to repair completion

PAYMENT:
- You pay your deductible directly to repair shop
- We pay remaining repair costs directly to shop
- If other party is at fault, we may waive your deductible"""
    },
    {
        "question": "Deductible Information",
        "content": """UNDERSTANDING YOUR DEDUCTIBLES

WHAT IS A DEDUCTIBLE?
A deductible is the amount you pay out-of-pocket before your insurance coverage kicks in. You pay the deductible directly to the repair shop, and insurance covers the rest.

YOUR DEDUCTIBLES:

COLLISION DEDUCTIBLE: $500
- Applies when: Your vehicle is damaged in a collision with another vehicle or object
- Example: Repair costs $3,000. You pay $500, insurance pays $2,500
- Per accident: You pay the deductible once per accident, not per vehicle

COMPREHENSIVE DEDUCTIBLE: $250
- Applies when: Damage from theft, vandalism, weather, falling objects, animal strikes
- Example: Windshield replacement costs $800. You pay $250, insurance pays $550
- Glass-only damage: $0 deductible for glass repair/replacement

WHEN YOU DON'T PAY A DEDUCTIBLE:
- Liability claims (damage to other party's property)
- Uninsured motorist claims (other driver at fault, no insurance)
- When other party's insurance pays your claim (at-fault accident by other driver)
- Glass repair/replacement (comprehensive coverage)

DEDUCTIBLE WAIVER PROGRAM:
- For each year without an at-fault claim, your deductible decreases by $50
- Maximum reduction: $200
- After 4 claim-free years, your collision deductible would be $300

CHANGING YOUR DEDUCTIBLE:
- You can adjust deductibles at policy renewal
- Higher deductible = Lower premium
- Lower deductible = Higher premium
- $500 collision deductible reduces premium by approximately $150/year vs $250 deductible"""
    },
    {
        "question": "Coverage Exclusions and Limitations",
        "content": """WHAT IS NOT COVERED

INTENTIONAL ACTS:
- Deliberate damage to your own vehicle
- Insurance fraud or false claims
- Damage caused intentionally by you or household members

RACING AND COMPETITIVE EVENTS:
- Racing, speed contests, or timed events
- Organized track days or autocross
- Preparation for racing activities

COMMERCIAL USE:
- Using personal vehicle for business deliveries (UberEats, DoorDash)
- Ridesharing (Uber, Lyft) without commercial endorsement
- Transportation of paying passengers
- Business use beyond commuting to work

IMPAIRED DRIVING:
- Driving under the influence of alcohol or drugs
- BAC above legal limit
- Driving while license is suspended or revoked

MECHANICAL BREAKDOWN:
- Engine failure due to wear and tear
- Transmission problems from normal use
- Brake wear
- Tire replacement (except from covered incident)
- Maintenance items (oil changes, filters, etc.)

PRIOR DAMAGE:
- Damage that existed before policy effective date
- Damage from undisclosed previous accidents

GOVERNMENT OR MILITARY ACTION:
- War, civil unrest, riots
- Nuclear hazard
- Government confiscation

CUSTOM EQUIPMENT LIMITATIONS:
- Aftermarket parts coverage limited to $1,000 unless specifically declared
- Sound system upgrades not covered beyond factory value
- Racing modifications void coverage

RENTAL VEHICLES:
- Damage to rental vehicles used in business
- Rentals outside the United States, Canada, and Mexico
- Rentals exceeding 30 consecutive days

TO ADD COVERAGE:
Contact us to discuss endorsements for:
- Commercial/rideshare use
- Custom equipment
- Extended rental coverage
- Gap insurance"""
    }
]

# Load existing docs
with open('insurance_docs.json') as f:
    existing_docs = json.load(f)

# Add sample policies at the beginning
combined_docs = sample_policies + existing_docs

# Save combined
with open('insurance_docs.json', 'w') as f:
    json.dump(combined_docs, f, indent=2)

print(f"✅ Added {len(sample_policies)} sample policy documents")
print(f"Total documents: {len(combined_docs)}")
print("\nSample policies added:")
for policy in sample_policies:
    print(f"  - {policy['question']}")