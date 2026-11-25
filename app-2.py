"""Customer, sales, profit, and returns dashboard. Run: streamlit run app.py"""
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Customer & Orders Analytics", page_icon="📊", layout="wide")
st.title("Customer & Orders Analytics")
st.caption("Explore sales, profitability, seasonality, and returns across 2023–2026.")
DATA_DIR = Path(__file__).resolve().parent

@st.cache_data(show_spinner=False)
def parse_csv_bytes(raw: bytes, filename: str):
    """Parse from the beginning every time, independent of upload cursor position."""
    from io import BytesIO
    last_error = None
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            frame = pd.read_csv(BytesIO(raw), encoding=encoding, dtype={"Postal Code": "string"})
            frame.columns = (frame.columns.astype(str)
                             .str.replace("\ufeff", "", regex=False)
                             .str.replace("ï»¿", "", regex=False).str.strip())
            return frame
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
    raise ValueError(f"Unable to parse {filename}: {last_error}")

def read_csv(source, filename):
    if hasattr(source, "getvalue"):
        # UploadedFile is a stream; reading it without resetting can start mid-file.
        # getvalue() always returns the entire uploaded file, irrespective of cursor.
        raw = source.getvalue()
    else:
        raw = Path(source).read_bytes()
    if not raw:
        raise ValueError(f"{filename} is empty. Re-upload the original CSV.")
    return parse_csv_bytes(raw, filename)

@st.cache_data
def prepare(customers, orders, returns):
    customers = customers.copy()
    orders = orders.copy()
    returns = returns.copy()
    expected = {
        "customers.csv": (customers, ["Customer ID", "Segment"]),
        "orders.csv": (orders, ["Order ID", "Customer ID", "Order Date", "Ship Date", "Category", "Sales", "Profit", "Quantity", "Discount"]),
        "returns.csv": (returns, ["Order ID", "Returned"]),
    }
    for filename, (table, required) in expected.items():
        missing = [col for col in required if col not in table.columns]
        if missing:
            raise ValueError(f"{filename} is missing columns: {missing}. Found columns: {table.columns.tolist()}. "
                             "The file may have been read from the middle or may not be the expected CSV. Clear the uploader and re-upload the original file.")
    for col in ("Order Date", "Ship Date"):
        orders[col] = pd.to_datetime(orders[col], errors="coerce", format="%m/%d/%Y")
    for col in ("Sales", "Profit", "Quantity", "Discount"):
        orders[col] = pd.to_numeric(orders[col], errors="coerce")
    # Customer locations vary for some IDs: retain only unambiguous segment lookup.
    segments = customers.groupby("Customer ID", as_index=False).agg(
        Segment=("Segment", lambda s: s.dropna().iloc[0] if s.dropna().nunique() == 1 else "Ambiguous")
    )
    merged = orders.merge(segments, on="Customer ID", how="left", validate="many_to_one", indicator=True)
    merged["Customer Matched"] = merged.pop("_merge").eq("both")
    returned_ids = set(returns.loc[returns["Returned"].astype(str).str.lower().eq("yes"), "Order ID"].dropna())
    merged["Returned"] = merged["Order ID"].isin(returned_ids)
    return merged

with st.sidebar:
    st.header("Data sources")
    st.caption("Bundled CSV files load by default. Upload a CSV only to replace a bundled file. If a previous upload was corrupted, remove it using the X beside the upload.")
    uploads = {name: st.file_uploader(f"{name.title()} CSV", type="csv", key=name) for name in ("customers", "orders", "returns")}

try:
    tables = {}
    for name, uploaded in uploads.items():
        path = uploaded if uploaded is not None else DATA_DIR / f"{name}.csv"
        if uploaded is None and not path.exists():
            st.error(f"Missing {name}.csv. Upload it in the sidebar or place it beside app.py.")
            st.stop()
        tables[name] = read_csv(path, f"{name}.csv")
    df = prepare(tables["customers"], tables["orders"], tables["returns"])
    if df["Order Date"].notna().sum() == 0:
        raise ValueError("No valid Order Date values were found in orders.csv. Expected dates such as 11/8/2025.")
except Exception as exc:
    st.error(f"Could not load the data: {exc}")
    st.stop()

valid_dates = df["Order Date"].dropna()
with st.sidebar:
    st.header("Filters")
    date_selection = st.date_input("Order date range", value=(valid_dates.min().date(), valid_dates.max().date()),
                                   min_value=valid_dates.min().date(), max_value=valid_dates.max().date())
    if not isinstance(date_selection, (tuple, list)) or len(date_selection) != 2:
        st.info("Select both a start date and an end date.")
        st.stop()
    start, end = date_selection
    segments = st.multiselect("Customer segments", sorted(df["Segment"].dropna().unique()), default=sorted(df["Segment"].dropna().unique()))
    categories = st.multiselect("Product categories", sorted(df["Category"].dropna().unique()), default=sorted(df["Category"].dropna().unique()))
    st.caption("Category return rates count distinct orders per category; one order can belong to several categories.")

filtered = df.loc[
    df["Order Date"].between(pd.Timestamp(start), pd.Timestamp(end))
    & df["Segment"].isin(segments)
    & df["Category"].isin(categories)
].copy()
if filtered.empty:
    st.warning("No order lines match the current filters.")
    st.stop()

st.subheader("Overview")
a,b,c,d = st.columns(4)
a.metric("Total sales", f"${filtered['Sales'].sum():,.2f}")
b.metric("Total profit", f"${filtered['Profit'].sum():,.2f}")
c.metric("Unique orders", f"{filtered['Order ID'].nunique():,}")
d.metric("Returned orders", f"{filtered.loc[filtered['Returned'], 'Order ID'].nunique():,}")
st.caption(f"Filtered: {len(filtered):,} order lines | {start} to {end} | "
           f"Segments: {', '.join(segments)} | Categories: {', '.join(categories)}")

sales_tab, profit_tab, returns_tab, data_tab = st.tabs(["Sales trends", "Profit", "Returns", "Data & validation"])

with sales_tab:
    st.subheader("Monthly sales by product category")
    st.caption("Chart type: Multi-series line chart")
    monthly = filtered.assign(Month=filtered["Order Date"].dt.to_period("M").dt.to_timestamp())
    monthly = monthly.groupby(["Month", "Category"], as_index=False)["Sales"].sum()
    fig = px.line(monthly, x="Month", y="Sales", color="Category", markers=True,
                  labels={"Sales": "Monthly sales ($)", "Month": "Month"})
    show_q4 = st.checkbox("Shade Q4 periods (October–December)", value=True)
    if show_q4:
        for year in range(start.year, end.year + 1):
            fig.add_vrect(x0=f"{year}-10-01", x1=f"{year+1}-01-01", fillcolor="gray", opacity=0.13,
                          line_width=0, layer="below")
    fig.update_layout(hovermode="x unified", legend_title_text="Product category")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Monthly total sales")
    st.caption("Chart type: Line chart")
    total_monthly = monthly.groupby("Month", as_index=False)["Sales"].sum()
    st.plotly_chart(px.line(total_monthly, x="Month", y="Sales", markers=True,
                            labels={"Sales": "Monthly sales ($)"}), use_container_width=True)

with profit_tab:
    st.subheader("Profit distribution by category")
    st.caption("Chart type: Pie chart")
    cat_profit = filtered.groupby("Category", as_index=False)["Profit"].sum()
    if (cat_profit["Profit"] >= 0).all() and cat_profit["Profit"].sum() > 0:
        st.plotly_chart(px.pie(cat_profit, names="Category", values="Profit", hole=0,
                               title="Share of total category profit"), use_container_width=True)
    else:
        st.info("Some category profits are negative. A bar chart is more accurate than a pie chart for signed values.")
        st.plotly_chart(px.bar(cat_profit, x="Category", y="Profit"), use_container_width=True)
    st.subheader("Profit by customer segment")
    st.caption("Chart type: Horizontal bar chart (descending profit)")
    seg_profit = filtered.groupby("Segment", as_index=False)["Profit"].sum().sort_values("Profit", ascending=False)
    fig = px.bar(seg_profit, x="Profit", y="Segment", orientation="h", text="Profit",
                 labels={"Profit": "Total profit ($)"})
    fig.update_traces(texttemplate="$%{x:,.0f}", textposition="outside")
    fig.update_yaxes(categoryorder="array", categoryarray=seg_profit["Segment"].tolist()[::-1])
    st.plotly_chart(fig, use_container_width=True)

with returns_tab:
    st.subheader("Return rate by product category")
    st.caption("Chart type: Horizontal bar chart (highest return rate at top)")
    group = filtered.groupby("Category").agg(
        Orders=("Order ID", "nunique"),
        Returned_Orders=("Order ID", lambda s: s[filtered.loc[s.index, "Returned"]].nunique())
    ).reset_index()
    group["Return Rate (%)"] = group["Returned_Orders"] / group["Orders"] * 100
    group = group.sort_values("Return Rate (%)", ascending=False)
    fig = px.bar(group, x="Return Rate (%)", y="Category", orientation="h", text="Return Rate (%)")
    fig.update_traces(texttemplate="%{x:.2f}%", textposition="outside")
    fig.update_yaxes(categoryorder="array", categoryarray=group["Category"].tolist()[::-1])
    fig.update_xaxes(range=[0, max(1, group["Return Rate (%)"].max() * 1.2)])
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(group.rename(columns={"Returned_Orders": "Returned orders"}), hide_index=True, use_container_width=True)
    st.info("Returns are recorded at the order level, not the product-line level. "
            "A returned order containing multiple categories is counted as returned in each category.")

with data_tab:
    st.subheader("Join validation")
    st.write(f"Order lines: **{len(df):,}** | Customer matches: **{df['Customer Matched'].sum():,}** | "
             f"Unmatched: **{(~df['Customer Matched']).sum():,}**")
    st.write(f"Distinct orders: **{df['Order ID'].nunique():,}** | "
             f"Distinct returned orders in orders data: **{df.loc[df['Returned'], 'Order ID'].nunique():,}**")
    st.caption("Customer segment is joined using a one-row-per-customer lookup. "
               "Customer location fields are omitted because customers may have multiple locations.")
    source = st.selectbox("Preview table", ["Filtered merged orders", "Customers", "Orders", "Returns"])
    previews = {"Filtered merged orders": filtered, "Customers": tables["customers"],
                "Orders": tables["orders"], "Returns": tables["returns"]}
    preview = previews[source]
    st.dataframe(preview.head(5), use_container_width=True)
    st.subheader("Column summary")
    descriptions = {"Sales": "Sales amount for an order line", "Profit": "Profit or loss for an order line",
                    "Quantity": "Units in order line", "Discount": "Discount rate", "Order ID": "Order identifier",
                    "Customer ID": "Customer identifier", "Returned": "Order return indicator",
                    "Order Date": "Date the order was placed", "Category": "Product category",
                    "Segment": "Customer segment", "Postal Code": "Postal/ZIP code (identifier)"}
    stats = []
    for col in preview.columns:
        s = preview[col]
        numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and col not in ("Row ID", "Postal Code")
        stats.append({"Column": col, "Type": str(s.dtype), "Description": descriptions.get(col, col.replace("_", " ")),
                      "Missing": int(s.isna().sum()), "Unique": int(s.nunique(dropna=True)),
                      "Min": s.min() if numeric else None, "Max": s.max() if numeric else None,
                      "Mean": s.mean() if numeric else None, "Median": s.median() if numeric else None,
                      "Std dev": s.std() if numeric else None})
    st.dataframe(pd.DataFrame(stats), hide_index=True, use_container_width=True)
    st.download_button("Download filtered merged orders (CSV)", filtered.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_orders.csv", mime="text/csv")

invalid = df["Order Date"].isna().sum()
if invalid:
    st.caption(f"Data quality note: {invalid} order line(s) have missing or invalid order dates and are excluded from date-filtered charts.")
