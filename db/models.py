"""
db/models.py
SQLAlchemy ORM models — customers, products, sales, invoices, suppliers, expenses, T&C
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean, Date
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()


class Customer(Base):
    """Customer master data with credit info."""
    __tablename__ = "customers"
    id               = Column(Integer, primary_key=True)
    name             = Column(String(100), nullable=False)
    email            = Column(String(120), unique=True)
    phone            = Column(String(20))
    address          = Column(Text)
    city             = Column(String(50))
    credit_limit     = Column(Float, default=10000.0)
    outstanding_balance = Column(Float, default=0.0)
    status           = Column(String(20), default="active")   # active, inactive, vip
    joined_date      = Column(Date)
    created_at       = Column(DateTime, default=datetime.utcnow)
    sales            = relationship("Sale", back_populates="customer")
    invoices         = relationship("Invoice", back_populates="customer")


class Supplier(Base):
    """Supplier / vendor master."""
    __tablename__ = "suppliers"
    id             = Column(Integer, primary_key=True)
    name           = Column(String(100), nullable=False)
    contact_person = Column(String(100))
    email          = Column(String(120))
    phone          = Column(String(20))
    payment_terms  = Column(String(100))
    active         = Column(Boolean, default=True)
    products       = relationship("Product", back_populates="supplier")


class Product(Base):
    """Product catalog with pricing and inventory."""
    __tablename__ = "products"
    id             = Column(Integer, primary_key=True)
    name           = Column(String(200), nullable=False)
    sku            = Column(String(50), unique=True)
    category       = Column(String(50))
    unit_price     = Column(Float, nullable=False)
    cost_price     = Column(Float)
    stock_quantity = Column(Integer, default=0)
    reorder_level  = Column(Integer, default=10)
    supplier_id    = Column(Integer, ForeignKey("suppliers.id"), nullable=True)
    active         = Column(Boolean, default=True)
    supplier       = relationship("Supplier", back_populates="products")
    sale_items     = relationship("SaleItem", back_populates="product")


class Sale(Base):
    """Sales transaction header."""
    __tablename__ = "sales"
    id             = Column(Integer, primary_key=True)
    customer_id    = Column(Integer, ForeignKey("customers.id"), nullable=False)
    sale_date      = Column(DateTime, default=datetime.utcnow)
    total_amount   = Column(Float, nullable=False)
    discount       = Column(Float, default=0.0)
    tax            = Column(Float, default=0.0)
    net_amount     = Column(Float, nullable=False)
    payment_method = Column(String(30), default="cash")
    payment_status = Column(String(20), default="paid")
    notes          = Column(Text)
    customer       = relationship("Customer", back_populates="sales")
    items          = relationship("SaleItem", back_populates="sale")


class SaleItem(Base):
    """Line items within a sale."""
    __tablename__ = "sale_items"
    id         = Column(Integer, primary_key=True)
    sale_id    = Column(Integer, ForeignKey("sales.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    quantity   = Column(Integer, nullable=False)
    unit_price = Column(Float, nullable=False)
    subtotal   = Column(Float, nullable=False)
    sale       = relationship("Sale", back_populates="items")
    product    = relationship("Product", back_populates="sale_items")


class Invoice(Base):
    """Customer invoices (can link to scanned bill image)."""
    __tablename__ = "invoices"
    id             = Column(Integer, primary_key=True)
    invoice_number = Column(String(30), unique=True)
    customer_id    = Column(Integer, ForeignKey("customers.id"), nullable=False)
    issue_date     = Column(Date)
    due_date       = Column(Date)
    amount         = Column(Float, nullable=False)
    paid_amount    = Column(Float, default=0.0)
    status         = Column(String(20), default="pending")  # pending, paid, overdue, partial
    notes          = Column(Text)
    image_path     = Column(String(255))   # path to scanned bill
    customer       = relationship("Customer", back_populates="invoices")


class Expense(Base):
    """Operational expense records."""
    __tablename__ = "expenses"
    id           = Column(Integer, primary_key=True)
    category     = Column(String(50))
    description  = Column(String(200))
    amount       = Column(Float, nullable=False)
    expense_date = Column(Date)
    paid_by      = Column(String(100))
    created_at   = Column(DateTime, default=datetime.utcnow)


class TermsConditions(Base):
    """T&C / policy documents indexed into RAG vector store."""
    __tablename__ = "terms_conditions"
    id             = Column(Integer, primary_key=True)
    title          = Column(String(200), nullable=False)
    category       = Column(String(50))   # return_policy, warranty, credit, payment
    content        = Column(Text, nullable=False)
    version        = Column(String(10), default="1.0")
    effective_date = Column(Date)
    active         = Column(Boolean, default=True)
    created_at     = Column(DateTime, default=datetime.utcnow)
