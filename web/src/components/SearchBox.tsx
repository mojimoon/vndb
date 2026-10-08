import { useEffect, useRef, useState } from "react";

/**
 * Text search that owns its own text and reports it with `onCommit`: on Enter,
 * on blur, and after a pause in typing. Nothing is reported while an IME is
 * composing (Chinese / Japanese input), so a controlled value coming back from
 * the URL can never overwrite half-composed text ("x" "xi" "xia" -> "夏").
 */
export function SearchBox({
  value,
  onCommit,
  delay = 350,
  className,
  placeholder,
}: {
  value: string;
  onCommit: (q: string) => void;
  delay?: number;
  className?: string;
  placeholder?: string;
}) {
  const [text, setText] = useState(value);
  const composing = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const committed = useRef(value);

  // Follow outside changes (e.g. "clear filters") unless the user is mid-input.
  useEffect(() => {
    if (value !== committed.current && !composing.current) {
      committed.current = value;
      setText(value);
    }
  }, [value]);
  useEffect(() => () => clearTimeout(timer.current), []);

  const commit = (q: string) => {
    clearTimeout(timer.current);
    if (q === committed.current) return;
    committed.current = q;
    onCommit(q);
  };
  const schedule = (q: string) => {
    clearTimeout(timer.current);
    timer.current = setTimeout(() => commit(q), delay);
  };

  return (
    <input
      type="search"
      value={text}
      onChange={(e) => {
        setText(e.target.value);
        if (!composing.current) schedule(e.target.value);
      }}
      onCompositionStart={() => {
        composing.current = true;
        clearTimeout(timer.current);
      }}
      onCompositionEnd={(e) => {
        composing.current = false;
        setText(e.currentTarget.value);
        schedule(e.currentTarget.value);
      }}
      onKeyDown={(e) => {
        if (e.key === "Enter" && !composing.current) commit(e.currentTarget.value);
      }}
      onBlur={(e) => commit(e.currentTarget.value)}
      placeholder={placeholder}
      aria-label={placeholder}
      className={className}
    />
  );
}
