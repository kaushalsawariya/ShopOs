"""
db/seed.py
Seeds the database with realistic test data.
Run: python -m db.seed  OR  called automatically on first startup.
"""

import random
from datetime import date, datetime, timedelta
from sqlalchemy.orm import Session
from db.database import engine, init_db
from db.models import Customer, Product, Sale, SaleItem, Invoice, Supplier, Expense, TermsConditions


# ── Suppliers ────────────────────────────────────────────────────────────────
SUPPLIERS = [
    dict(name="TechSupply Co.",     contact_person="Ravi Sharma",  email="ravi@techsupply.com",   phone="9876543210", payment_terms="Net 30"),
    dict(name="Global Goods Ltd.",  contact_person="Priya Mehta",  email="priya@globalgoods.in",  phone="9812345678", payment_terms="Net 45"),
    dict(name="FastDeliver Inc.",   contact_person="Ankit Singh",  email="ankit@fastdeliver.com", phone="9898765432", payment_terms="Net 15"),
    dict(name="Reliable Parts Pvt.",contact_person="Deepa Nair",   email="deepa@reliable.co",     phone="9765432109", payment_terms="Net 30"),
    dict(name="Mega Wholesale",     contact_person="Arjun Bose",   email="arjun@mega.in",         phone="9654321098", payment_terms="COD"),
]

# ── Products ─────────────────────────────────────────────────────────────────
PRODUCTS = [
    dict(name="Laptop 15\" i5",            sku="TECH-001", category="Electronics",    unit_price=55000, cost_price=45000, stock_quantity=25, reorder_level=5),
    dict(name="Wireless Mouse",             sku="TECH-002", category="Electronics",    unit_price=899,   cost_price=450,   stock_quantity=150,reorder_level=20),
    dict(name="Mechanical Keyboard",        sku="TECH-003", category="Electronics",    unit_price=3500,  cost_price=2200,  stock_quantity=80, reorder_level=15),
    dict(name="USB-C Hub 7-in-1",           sku="TECH-004", category="Electronics",    unit_price=2200,  cost_price=1200,  stock_quantity=60, reorder_level=10),
    dict(name="Monitor 24\" FHD",           sku="TECH-005", category="Electronics",    unit_price=15000, cost_price=11000, stock_quantity=30, reorder_level=5),
    dict(name="Office Chair Ergonomic",     sku="FURN-001", category="Furniture",      unit_price=8500,  cost_price=5500,  stock_quantity=20, reorder_level=3),
    dict(name="Standing Desk 140cm",        sku="FURN-002", category="Furniture",      unit_price=22000, cost_price=15000, stock_quantity=10, reorder_level=2),
    dict(name="LED Desk Lamp",              sku="FURN-003", category="Furniture",      unit_price=1200,  cost_price=700,   stock_quantity=100,reorder_level=15),
    dict(name="A4 Paper Ream 500 sheets",   sku="STAT-001", category="Stationery",     unit_price=350,   cost_price=200,   stock_quantity=500,reorder_level=50),
    dict(name="Ballpoint Pens Box of 20",   sku="STAT-002", category="Stationery",     unit_price=120,   cost_price=60,    stock_quantity=300,reorder_level=40),
    dict(name="Heavy Duty Stapler",         sku="STAT-003", category="Stationery",     unit_price=450,   cost_price=250,   stock_quantity=75, reorder_level=10),
    dict(name="Printer Ink Cartridge",      sku="STAT-004", category="Stationery",     unit_price=1800,  cost_price=1100,  stock_quantity=120,reorder_level=20),
    dict(name="Webcam 1080p HD",            sku="TECH-006", category="Electronics",    unit_price=3200,  cost_price=2000,  stock_quantity=45, reorder_level=8),
    dict(name="Noise Cancelling Headphones",sku="TECH-007", category="Electronics",    unit_price=7500,  cost_price=5000,  stock_quantity=35, reorder_level=5),
    dict(name="External SSD 1TB",           sku="TECH-008", category="Electronics",    unit_price=9500,  cost_price=7000,  stock_quantity=40, reorder_level=8),
    dict(name="Whiteboard 4x3 ft",          sku="FURN-004", category="Furniture",      unit_price=3500,  cost_price=2000,  stock_quantity=15, reorder_level=3),
    dict(name="Filing Cabinet 3-Drawer",    sku="FURN-005", category="Furniture",      unit_price=6500,  cost_price=4200,  stock_quantity=12, reorder_level=2),
    dict(name="Hand Sanitizer 500ml",       sku="HLTH-001", category="Health & Safety",unit_price=180,   cost_price=80,    stock_quantity=200,reorder_level=30),
    dict(name="First Aid Kit Complete",     sku="HLTH-002", category="Health & Safety",unit_price=950,   cost_price=550,   stock_quantity=30, reorder_level=5),
    dict(name="Power Extension 6-Socket",   sku="TECH-009", category="Electronics",    unit_price=850,   cost_price=500,   stock_quantity=90, reorder_level=15),
]

# ── Customers ─────────────────────────────────────────────────────────────────
CUSTOMERS = [
    dict(name="Akash Enterprises",   email="akash@akashent.com",   phone="9876501234", city="Delhi",    credit_limit=100000, outstanding_balance=15000, status="vip",      joined_date=date(2021,3,15)),
    dict(name="Sunita Retail Hub",   email="sunita@retail.in",     phone="9845678901", city="Delhi",    credit_limit=50000,  outstanding_balance=8000,  status="active",   joined_date=date(2022,1,10)),
    dict(name="Patel Brothers",      email="patel@brothers.co",    phone="9823456789", city="Delhi",    credit_limit=75000,  outstanding_balance=0,     status="active",   joined_date=date(2020,6,20)),
    dict(name="Meena Tech Solutions",email="meena@meenatech.com",  phone="9811234567", city="Gurugram", credit_limit=200000, outstanding_balance=45000, status="vip",      joined_date=date(2019,11,5)),
    dict(name="Kapoor Stationery",   email="kapoor@stationery.in", phone="9899876543", city="Delhi",    credit_limit=30000,  outstanding_balance=0,     status="active",   joined_date=date(2023,2,14)),
    dict(name="Rajesh Office Supply",email="rajesh@officesup.com", phone="9876123456", city="Faridabad",credit_limit=40000,  outstanding_balance=12000, status="active",   joined_date=date(2022,8,22)),
    dict(name="Verma Constructions", email="verma@construct.net",  phone="9812987654", city="Gurugram", credit_limit=150000, outstanding_balance=0,     status="vip",      joined_date=date(2021,7,30)),
    dict(name="Sharma Medical",      email="sharma@medical.org",   phone="9823654321", city="Delhi",    credit_limit=60000,  outstanding_balance=22000, status="active",   joined_date=date(2022,4,18)),
    dict(name="Gupta Traders",       email="gupta@traders.co.in",  phone="9867543210", city="Delhi",    credit_limit=25000,  outstanding_balance=0,     status="inactive", joined_date=date(2020,12,1)),
    dict(name="Nova IT Park",        email="nova@itpark.in",       phone="9890123456", city="Noida",    credit_limit=500000, outstanding_balance=75000, status="vip",      joined_date=date(2018,5,10)),
]

# ── Terms & Conditions documents (also used by RAG to understand bills) ──────
TERMS = [
    dict(
        title="Return and Refund Policy",
        category="return_policy",
        version="2.1",
        effective_date=date(2024,1,1),
        content="""RETURN AND REFUND POLICY — Effective January 1, 2024

1. ELIGIBILITY FOR RETURNS
   Items may be returned within 30 days of purchase for a full refund provided the item is:
   - In original unused condition with all packaging intact
   - Accompanied by original invoice or receipt
   - Not in the excluded categories listed below

2. NON-RETURNABLE ITEMS
   - Software licenses and digital products
   - Opened consumables (paper, ink cartridges)
   - Health and safety products (sanitizers, first aid kits)
   - Items damaged due to customer misuse
   - Custom or specially sourced orders

3. REFUND TIMELINE
   - Cash purchases: refund within 3 business days
   - Card purchases: 5–7 business days to original card
   - Credit account customers: credited to account within 2 business days

4. EXCHANGE POLICY
   Items eligible for return may be exchanged for items of equal or greater value.
   Price difference will be charged or credited accordingly.

5. WARRANTY CLAIMS
   Warranty defects are handled under the Warranty Policy section.
   Please refer to product documentation for warranty duration and coverage details.
"""),
    dict(
        title="Credit Account Terms",
        category="credit",
        version="1.3",
        effective_date=date(2023,6,1),
        content="""CREDIT ACCOUNT TERMS AND CONDITIONS

1. CREDIT FACILITY
   Credit accounts are available to registered business customers subject to approval.
   Credit limits are assigned based on business volume, payment history, and creditworthiness.

2. PAYMENT TERMS
   Standard terms: Net 30 days from invoice date.
   VIP accounts: Net 45 days from invoice date.
   Early payment discount: 2% if paid within 10 days (2/10 Net 30).

3. OVERDUE ACCOUNTS
   - 15 days overdue: reminder notice sent
   - 30 days overdue: credit facility suspended
   - Interest of 2% per month charged on amounts overdue beyond 30 days
   - 60+ days overdue: referred for collection

4. ACCOUNT SUSPENSION
   Credit may be suspended for: consistent late payments, returned cheques, fraud,
   or violation of purchase terms.

5. DISPUTE RESOLUTION
   Billing disputes must be raised within 15 days of invoice receipt.
"""),
    dict(
        title="Warranty Policy",
        category="warranty",
        version="1.0",
        effective_date=date(2024,1,1),
        content="""WARRANTY POLICY

1. STANDARD WARRANTY PERIODS
   - Electronics: 1-year manufacturer warranty
   - Furniture: 2-year structural warranty
   - Stationery: No warranty (unless defective at purchase)

2. COVERAGE
   Covers manufacturing defects under normal usage.
   Does NOT cover: physical damage, water damage, unauthorised repair,
   normal wear and tear, or consumable parts.

3. CLAIM PROCESS
   Step 1: Contact service team with proof of purchase
   Step 2: Product inspection within 3 business days
   Step 3: Repair, replacement, or refund decision within 7 business days

4. EXTENDED WARRANTY
   Available for electronics at time of purchase:
   - 1-year extension: 8% of product price
   - 2-year extension: 15% of product price

5. ON-SITE WARRANTY
   Orders above ₹50,000 on a single invoice qualify for on-site warranty service
   delivered within 48 hours of claim registration.
"""),
    dict(
        title="Payment and Pricing Policy",
        category="payment",
        version="1.5",
        effective_date=date(2024,3,1),
        content="""PAYMENT AND PRICING POLICY

1. ACCEPTED PAYMENT METHODS
   Cash, Credit/Debit cards (Visa, Mastercard, RuPay), UPI (GPay, PhonePe, Paytm),
   NEFT/RTGS/IMPS bank transfers, approved credit accounts, and cheques (subject to clearance).

2. PRICING
   All prices include GST unless otherwise stated.
   Prices subject to change without prior notice.
   Invoice price prevails over catalogue price in case of discrepancy.

3. DISCOUNT STRUCTURE
   - Orders above ₹25,000: 5% discount
   - Orders above ₹1,00,000: 8% discount
   - VIP customers receive additional 3% on all purchases
   - Seasonal promotions cannot be combined with other discounts

4. GST RATES
   - Electronics: 18% GST
   - Furniture: 12% GST
   - Stationery: 12% GST
   - Health & Safety products: 5% GST

5. FAILED PAYMENTS
   Bounced cheques incur ₹500 penalty per instance.
   Repeated failures may result in withdrawal of credit facilities.
"""),
    dict(
        title="Bill and Invoice Reading Guide",
        category="billing",
        version="1.0",
        effective_date=date(2024,1,1),
        content="""BILL AND INVOICE UNDERSTANDING GUIDE

1. INVOICE STRUCTURE
   Every invoice issued by this shop contains:
   - Invoice Number (format: INV-YYYY-NNNN)
   - Issue Date and Due Date
   - Customer name and billing address
   - Itemised list of products with quantity, unit price, and subtotal
   - Subtotal before tax
   - GST breakdown by category
   - Total amount payable
   - Payment terms and accepted payment methods
   - Late payment charges if applicable

2. READING ITEMISED BILLS
   Each line item shows: SKU code, product description, quantity, unit price, GST rate, and line total.
   Discount lines appear after the subtotal if any discount was applied.
   GST is itemised per applicable rate (5%, 12%, or 18%).

3. PAYMENT STATUS INDICATORS
   PAID: Full payment received, no balance due.
   PENDING: Invoice issued, payment not yet received.
   PARTIAL: Some payment received, balance outstanding.
   OVERDUE: Due date passed, payment not received.

4. COMMON ABBREVIATIONS ON BILLS
   GST = Goods and Services Tax
   CGST = Central GST (half of total GST)
   SGST = State GST (half of total GST)
   IGST = Integrated GST (for inter-state transactions)
   PO = Purchase Order
   LPO = Local Purchase Order
   TDS = Tax Deducted at Source

5. DISPUTE PROCESS FOR BILLS
   If you believe a bill contains an error, contact accounts within 7 days.
   Include the invoice number, the disputed line item, and the reason for dispute.
   We will investigate and issue a corrected invoice or credit note within 5 business days.
"""),
]


def seed_database():
    """Seeds all tables with test data. Skips if data already exists."""
    init_db()
    with Session(engine) as db:
        if db.query(Customer).count() > 0:
            return   # Already seeded

        # Suppliers
        sup_objs = [Supplier(**s) for s in SUPPLIERS]
        db.add_all(sup_objs)
        db.flush()

        # Products (assign supplier round-robin)
        prod_objs = []
        for i, p in enumerate(PRODUCTS):
            prod = Product(**p, supplier_id=sup_objs[i % len(sup_objs)].id)
            db.add(prod)
            prod_objs.append(prod)
        db.flush()

        # Customers
        cust_objs = [Customer(**c) for c in CUSTOMERS]
        db.add_all(cust_objs)
        db.flush()

        # Sales + line items (35 realistic transactions)
        rng = random.Random(42)   # deterministic seed for reproducibility
        methods = ["cash", "card", "upi", "bank_transfer", "credit"]
        statuses = ["paid", "paid", "paid", "pending", "partial"]
        start = date(2024, 1, 1)

        for i in range(35):
            cust = rng.choice(cust_objs)
            prods = rng.sample(prod_objs, rng.randint(1, 5))
            s_date = datetime.combine(start + timedelta(days=rng.randint(0, 364)), datetime.min.time())

            total = sum(p.unit_price * rng.randint(1, 5) for p in prods)
            disc  = round(total * rng.choice([0, 0, 0.05, 0.08]), 2)
            tax   = round((total - disc) * 0.18, 2)
            net   = round(total - disc + tax, 2)

            sale = Sale(
                customer_id=cust.id, sale_date=s_date,
                total_amount=round(total,2), discount=disc, tax=tax, net_amount=net,
                payment_method=rng.choice(methods), payment_status=rng.choice(statuses),
            )
            db.add(sale)
            db.flush()

            for p in prods:
                qty = rng.randint(1, 5)
                db.add(SaleItem(sale_id=sale.id, product_id=p.id,
                                quantity=qty, unit_price=p.unit_price,
                                subtotal=round(p.unit_price*qty, 2)))

        # Invoices (2–4 per customer)
        for i, cust in enumerate(cust_objs):
            for j in range(rng.randint(2, 4)):
                iss = start + timedelta(days=rng.randint(0, 300))
                due = iss + timedelta(days=30)
                amt = round(rng.uniform(5000, 80000), 2)
                paid = round(rng.choice([amt, amt*0.5, 0]), 2)
                st = "paid" if paid >= amt else ("partial" if paid > 0 else "pending")
                if due < date.today() and st != "paid":
                    st = "overdue"
                db.add(Invoice(
                    invoice_number=f"INV-2024-{i*10+j+1:04d}",
                    customer_id=cust.id, issue_date=iss, due_date=due,
                    amount=amt, paid_amount=paid, status=st,
                ))

        # Expenses (10 categories × 6 months)
        expense_data = [
            ("rent",        "Monthly shop rent",          45000),
            ("utilities",   "Electricity bill",            8500),
            ("utilities",   "Internet and phone",          3200),
            ("salaries",    "Staff salaries (3 people)",  75000),
            ("maintenance", "Repair and maintenance",     15000),
            ("transport",   "Delivery and logistics",      6500),
            ("marketing",   "Digital advertising",         4000),
            ("insurance",   "Shop insurance premium",     12000),
            ("misc",        "Office sundry expenses",      2500),
            ("utilities",   "Water charges",               1200),
        ]
        for cat, desc, amt in expense_data:
            for month in range(1, 7):
                db.add(Expense(
                    category=cat, description=desc,
                    amount=amt + rng.randint(-500, 500),
                    expense_date=date(2024, month, rng.randint(1, 28)),
                    paid_by="Shop Owner",
                ))

        # Terms & Conditions
        for t in TERMS:
            db.add(TermsConditions(**t))

        db.commit()
