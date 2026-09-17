import { severityLabel } from '../lib/format';

export function SeverityTag({ severity }: { severity: string }) {
  return (
    <span className="sev" data-sev={severity}>
      {severityLabel(severity)}
    </span>
  );
}
