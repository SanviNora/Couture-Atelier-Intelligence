-- One row per client: home store (most visited), visit history, breadth.
-- IDs with 10+ receipts or 4+ stores on a single day are shared or promotional
-- cards, not people; they are flagged so client models can exclude them.
create or replace table marts.client_profile as
with daily as (
    select client_id, purchase_date,
           count(distinct receipt_id) as receipts, count(distinct store_id) as stores
    from marts.fct_purchase
    group by all
),
peaks as (
    select client_id, max(receipts) as max_receipts_per_day, max(stores) as max_stores_per_day
    from daily
    group by 1
),
store_visits as (
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
    any_value(k.max_receipts_per_day) as max_receipts_per_day,
    any_value(k.max_stores_per_day) as max_stores_per_day,
    any_value(k.max_receipts_per_day) >= 10 or any_value(k.max_stores_per_day) >= 4 as is_shared_card,
    'derived' as data_provenance
from marts.fct_purchase p
join home h using (client_id)
join peaks k using (client_id)
group by p.client_id, h.home_store_id;
