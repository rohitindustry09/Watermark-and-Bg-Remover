import { useId, type ReactNode } from 'react'

export function Section({ title, children, aside }: { title: string; children: ReactNode; aside?: ReactNode }) {
  return (
    <section className="border-b border-line px-3 py-3">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted">{title}</h3>
        {aside}
      </div>
      <div className="space-y-2.5">{children}</div>
    </section>
  )
}

interface SliderProps {
  label: string
  value: number
  min: number
  max: number
  step?: number
  unit?: string
  onChange: (v: number) => void
  onReset?: () => void
  hint?: string
}

export function Slider({ label, value, min, max, step = 1, unit = '', onChange, onReset, hint }: SliderProps) {
  const id = useId()
  return (
    <div title={hint}>
      <div className="flex items-baseline justify-between">
        <label htmlFor={id} className="text-fg">
          {label}
        </label>
        <span className="flex items-center gap-1.5">
          <output className="font-mono text-[12px] text-muted">
            {Number.isInteger(step) ? value : value.toFixed(2)}
            {unit}
          </output>
          {onReset && (
            <button
              type="button"
              onClick={onReset}
              aria-label={`Reset ${label}`}
              className="rounded px-1 text-[11px] text-muted hover:text-fg"
            >
              ↺
            </button>
          )}
        </span>
      </div>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </div>
  )
}

export function Toggle({ label, checked, onChange, hint }: { label: string; checked: boolean; onChange: (v: boolean) => void; hint?: string }) {
  const id = useId()
  return (
    <label htmlFor={id} className="flex cursor-pointer items-center justify-between gap-3" title={hint}>
      <span>{label}</span>
      <input id={id} type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="h-4 w-4 accent-[var(--color-accent)]" />
    </label>
  )
}

export function Btn({
  children,
  onClick,
  variant = 'ghost',
  disabled,
  title,
  pressed,
  className = '',
}: {
  children: ReactNode
  onClick?: () => void
  variant?: 'primary' | 'ghost' | 'danger'
  disabled?: boolean
  title?: string
  pressed?: boolean
  className?: string
}) {
  const base = 'inline-flex items-center justify-center gap-1.5 rounded-md px-2.5 py-1.5 text-[12.5px] font-medium transition-colors select-none'
  const look =
    variant === 'primary'
      ? 'bg-accent text-accent-ink hover:brightness-110'
      : variant === 'danger'
        ? 'border border-line text-danger hover:bg-raised'
        : pressed
          ? 'border border-accent/70 bg-raised text-fg'
          : 'border border-line text-fg hover:bg-raised'
  return (
    <button type="button" onClick={onClick} disabled={disabled} title={title} aria-pressed={pressed} className={`${base} ${look} ${className}`}>
      {children}
    </button>
  )
}

export function Select<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: T
  options: { value: T; label: string }[]
  onChange: (v: T) => void
}) {
  const id = useId()
  return (
    <div>
      <label htmlFor={id} className="mb-1 block text-muted">
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
        className="w-full rounded-md border border-line bg-raised px-2 py-1.5 text-fg"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  )
}

export function ColorField({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  const id = useId()
  return (
    <div className="flex items-center justify-between gap-3">
      <label htmlFor={id}>{label}</label>
      <span className="flex items-center gap-2">
        <span className="font-mono text-[12px] text-muted">{value}</span>
        <input id={id} type="color" value={value} onChange={(e) => onChange(e.target.value)} className="h-6 w-8 cursor-pointer rounded border border-line bg-transparent p-0" />
      </span>
    </div>
  )
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T
  options: { value: T; label: string; title?: string }[]
  onChange: (v: T) => void
  label: string
}) {
  return (
    <div role="group" aria-label={label} className="flex rounded-md border border-line p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          title={o.title}
          aria-pressed={o.value === value}
          onClick={() => onChange(o.value)}
          className={`flex-1 rounded px-2 py-1 text-[12px] ${o.value === value ? 'bg-raised text-fg' : 'text-muted hover:text-fg'}`}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}
