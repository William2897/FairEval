export const Loader2 = ({ size, className }: { size?: number; className?: string }) => (
  <div data-testid="loader-icon" style={{ width: size, height: size }} className={className}>
    Loading...
  </div>
);

export const AlertCircle = ({ size, className }: { size?: number; className?: string }) => (
  <div data-testid="alert-icon" style={{ width: size, height: size }} className={className}>
    Alert
  </div>
);

export const Info = ({ size, className }: { size?: number; className?: string }) => (
  <div data-testid="info-icon" style={{ width: size, height: size }} className={className}>
    Info
  </div>
);