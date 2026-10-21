-- One row per client purchase line. Lines from the same client at the same
-- timestamp form one receipt (a single visit).
create or replace table marts.fct_purchase as
select
    client_id,
    store_id,
    product_id,
    purchased_at,
    cast(purchased_at as date) as purchase_date,
    qty,
    dense_rank() over (order by client_id, purchased_at) as receipt_id,
    'observed_public' as data_provenance
from raw.purchases;
