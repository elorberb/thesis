-- Backfill profiles for any auth users created before the handle_new_user trigger
-- existed (their profile row was never created). Idempotent and safe to re-run.
insert into profiles (id, full_name)
select u.id, coalesce(u.raw_user_meta_data ->> 'full_name', u.raw_user_meta_data ->> 'name', '')
from auth.users u
left join profiles p on p.id = u.id
where p.id is null
on conflict (id) do nothing;
