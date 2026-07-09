-- Allow users to update their own analyses (corrections, patch)
create policy "users can update their own analyses"
    on analyses for update
    using (auth.uid() = performed_by);

-- Allow users to delete their own analyses
create policy "users can delete their own analyses"
    on analyses for delete
    using (auth.uid() = performed_by);
