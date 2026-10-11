# Task: analyze

Identify up to $max_findings causal findings for the incident, using only codes from the
taxonomy. For each finding give a confidence from 0 to 1, the ids of the retrieved
passages that support it, and one sentence of rationale. Return an empty list if the
evidence supports no finding.

## Incident packet

$packet

## Retrieved passages

$passages

## Reviewer feedback on your previous answer

$feedback
