-- Store x product x week for the first 12 weeks after launch.
-- Weekly sales in the source are net of returns, so they are split into
-- units sold and units returned. Restocks are matched on the ISO week start.
create or replace table marts.fct_store_product_week as
with units as (
    select *
    from (select product_id, store_id, release_date, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11 from raw.sales)
    unpivot include nulls (net_units for week_label in (w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11))
),
discounts as (
    select *
    from (select product_id, store_id, price_index, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11 from raw.price_discount)
    unpivot include nulls (discount for week_label in (d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11))
),
restock_weeks as (
    select
        product_id,
        store_id,
        cast(date_trunc('week', make_date(year, 1, 4)) + to_weeks(week - 1) as date) as week_start,
        sum(qty) as restock_units
    from raw.restocks
    group by all
)
select
    u.product_id,
    u.store_id,
    cast(replace(u.week_label, 'w', '') as integer) as week_index,
    cast(u.release_date + to_weeks(cast(replace(u.week_label, 'w', '') as integer)) as date) as week_start,
    greatest(u.net_units, 0) as units_sold,
    greatest(-u.net_units, 0) as units_returned,
    u.net_units,
    coalesce(d.discount, 0) as discount,
    d.discount is null as discount_missing,
    d.price_index,
    d.price_index * (1 - coalesce(d.discount, 0)) as price_paid_index,
    coalesce(d.discount, 0) > 0 as on_discount,
    coalesce(r.restock_units, 0) as restock_units,
    'observed_public' as data_provenance
from units u
join discounts d
    on d.product_id = u.product_id
    and d.store_id = u.store_id
    and replace(d.week_label, 'd', '') = replace(u.week_label, 'w', '')
left join restock_weeks r
    on r.product_id = u.product_id
    and r.store_id = u.store_id
    and r.week_start = cast(u.release_date + to_weeks(cast(replace(u.week_label, 'w', '') as integer)) as date);
