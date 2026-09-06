export function ThinkingPlaceholder() {
  return <div className="thinking-placeholder" role="status">
    <span className="thinking-placeholder__dots" aria-hidden="true"><i /><i /><i /></span>
    <span>等待回应</span>
  </div>;
}
