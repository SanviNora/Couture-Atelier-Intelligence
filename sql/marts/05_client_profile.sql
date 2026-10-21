-- One row per client: home store (most visited), visit history, breadth.
create or replace table marts.client_profile as
with store_visits as (
    select client_id, store_id, count(distinct receipt_id) as visits, max(purchased_at) as last_at
    from marts.fct_purchase
    group by all
),
home as (
    select client_id, store_id as home_store_id
    from store_visits
    qualify row_number() over (partition by client_id order by visits desc, last_at desc, store_id) = 1
)
select
    p.client_id,
    h.home_store_id,
    min(p.purchased_at) as first_purchase_at,
    max(p.purchased_at) as last_purchase_at,
    count(distinct p.receipt_id) as visits,
    sum(p.qty) as units,
    count(distinct p.product_id) as distinct_products,
    count(distinct p.store_id) as stores_visited,
    'derived' as data_provenance
from marts.fct_purchase p
join home h using (client_id)
group by p.client_id, h.home_store_id;
