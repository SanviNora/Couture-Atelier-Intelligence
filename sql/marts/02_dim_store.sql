-- One row per store, with observed volume used later to assign store formats.
create or replace table marts.dim_store as
with panel as (
    select
        store_id,
        count(distinct product_id) as products_stocked,
        sum(greatest(w0, 0) + greatest(w1, 0) + greatest(w2, 0) + greatest(w3, 0)
            + greatest(w4, 0) + greatest(w5, 0) + greatest(w6, 0) + greatest(w7, 0)
            + greatest(w8, 0) + greatest(w9, 0) + greatest(w10, 0) + greatest(w11, 0)) as panel_units
    from raw.sales
    group by store_id
),
clients as (
    select
        store_id,
        count(distinct client_id) as clients,
        sum(qty) as client_units,
        min(purchased_at) as first_purchase_at,
        max(purchased_at) as last_purchase_at
    from raw.purchases
    group by store_id
),
stores as (
    select store_id from panel
    union
    select store_id from clients
)
select
    s.store_id,
    coalesce(p.products_stocked, 0) as products_stocked,
    coalesce(p.panel_units, 0) as panel_units,
    coalesce(c.clients, 0) as clients,
    coalesce(c.client_units, 0) as client_units,
    c.first_purchase_at,
    c.last_purchase_at,
    p.store_id is not null as in_sales_panel,
    'derived' as data_provenance
from stores s
left join panel p using (store_id)
left join clients c using (store_id);
