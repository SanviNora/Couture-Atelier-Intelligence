-- One row per product. Attributes come from the sales panel; products seen only
-- in client purchases are kept with has_attributes = false.
create or replace table marts.dim_product as
with attributed as (
    select
        s.product_id,
        any_value(s.season) as season,
        any_value(s.category) as category,
        any_value(s.color) as color,
        any_value(s.fabric) as fabric,
        any_value(s.image_path) as image_path,
        min(s.release_date) as release_date,
        avg(p.price_index) as price_index,
        count(distinct s.store_id) as stores_launched
    from raw.sales s
    left join raw.price_discount p using (product_id, store_id)
    group by s.product_id
),
purchase_only as (
    select distinct product_id
    from raw.purchases
    where product_id not in (select product_id from attributed)
)
select *, true as has_attributes, 'observed_public' as data_provenance from attributed
union all
select
    product_id, null, null, null, null, null, null, null, 0,
    false, 'observed_public'
from purchase_only;
