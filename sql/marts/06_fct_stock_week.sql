-- Reconstructed stock per store x product x week for the 12 weeks after launch.
-- Opening stock = deliveries logged before launch; each week adds restocks and
-- returns and removes units sold. A week is "censored" when the store sold out
-- during it (sales <= what clients wanted) and "unavailable" when it had no
-- stock at the start. Visuelle has no stock table, so this is an estimate:
-- supply covers sales in about 99% of weeks; the rest are flagged.
create or replace table marts.fct_stock_week as
with opening as (
    select r.product_id, r.store_id, sum(r.qty) as opening_units
    from raw.restocks r
    join raw.sales s using (product_id, store_id)
    where cast(date_trunc('week', make_date(r.year, 1, 4)) + to_weeks(r.week - 1) as date) < s.release_date
    group by all
),
flows as (
    select w.*, coalesce(o.opening_units, 0) as opening_units,
           sum(w.restock_units + w.units_returned - w.units_sold) over (
               partition by w.product_id, w.store_id order by w.week_index
               rows between unbounded preceding and 1 preceding) as net_before
    from marts.fct_store_product_week w
    left join opening o using (product_id, store_id)
),
stock as (
    select *,
           opening_units + coalesce(net_before, 0) + restock_units + units_returned as stock_available,
           opening_units + coalesce(net_before, 0) + restock_units + units_returned - units_sold as stock_end
    from flows
)
select
    product_id, store_id, week_index, week_start, units_sold, units_returned, restock_units,
    discount, on_discount, price_index, opening_units,
    stock_available, stock_end,
    stock_available <= 0 as unavailable,
    stock_available > 0 and stock_end <= 0 as sold_out,
    stock_available <= 0 or stock_end <= 0 as censored,
    stock_end < 0 as supply_gap,
    'derived' as data_provenance
from stock;
