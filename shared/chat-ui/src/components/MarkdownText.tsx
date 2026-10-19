import "@assistant-ui/react-markdown/styles/dot.css";

import {
  type CodeHeaderProps,
  MarkdownTextPrimitive,
  memoizeMarkdownComponents,
  useIsMarkdownCodeBlock,
} from "@assistant-ui/react-markdown";
import { CheckIcon, CopyIcon } from "lucide-react";
import { type FC, memo, useCallback, useState } from "react";
import remarkGfm from "remark-gfm";

import { cn } from "../lib/cn";

const useCopyToClipboard = (resetMs = 2500) => {
  const [isCopied, setIsCopied] = useState(false);
  const copy = useCallback(
    (value: string) => {
      if (!value) return;
      void navigator.clipboard.writeText(value).then(() => {
        setIsCopied(true);
        window.setTimeout(() => setIsCopied(false), resetMs);
      });
    },
    [resetMs],
  );
  return { isCopied, copy };
};

const CodeHeader: FC<CodeHeaderProps> = ({ language, code }) => {
  const { isCopied, copy } = useCopyToClipboard();
  return (
    <div className="aui-code-header-root mt-3 flex items-center justify-between rounded-t-xl border border-b-0 border-border/60 bg-muted/60 px-3.5 py-1.5 text-xs">
      <span className="aui-code-header-language font-medium lowercase text-muted-foreground">{language}</span>
      <button
        type="button"
        onClick={() => copy(code ?? "")}
        className="rounded-md p-1 text-muted-foreground hover:bg-accent hover:text-foreground"
        aria-label="Copy code"
      >
        {isCopied ? <CheckIcon className="size-4" /> : <CopyIcon className="size-4" />}
      </button>
    </div>
  );
};

const defaultComponents = memoizeMarkdownComponents({
  h1: ({ className, ...props }) => (
    <h1 className={cn("aui-md-h1 mt-5 mb-2 text-2xl font-semibold first:mt-0 last:mb-0", className)} {...props} />
  ),
  h2: ({ className, ...props }) => (
    <h2 className={cn("aui-md-h2 mt-5 mb-2 text-xl font-semibold first:mt-0 last:mb-0", className)} {...props} />
  ),
  h3: ({ className, ...props }) => (
    <h3 className={cn("aui-md-h3 mt-4 mb-1.5 text-lg font-semibold first:mt-0 last:mb-0", className)} {...props} />
  ),
  h4: ({ className, ...props }) => (
    <h4 className={cn("aui-md-h4 mt-3.5 mb-1 text-base font-semibold first:mt-0 last:mb-0", className)} {...props} />
  ),
  h5: ({ className, ...props }) => (
    <h5 className={cn("aui-md-h5 mt-3 mb-1 text-base font-medium first:mt-0 last:mb-0", className)} {...props} />
  ),
  h6: ({ className, ...props }) => (
    <h6 className={cn("aui-md-h6 mt-3 mb-1 text-base font-medium first:mt-0 last:mb-0", className)} {...props} />
  ),
  p: ({ className, ...props }) => (
    <p className={cn("aui-md-p my-3 leading-relaxed first:mt-0 last:mb-0", className)} {...props} />
  ),
  a: ({ className, ...props }) => (
    <a className={cn("aui-md-a text-primary underline underline-offset-2 hover:text-primary/80", className)} {...props} />
  ),
  blockquote: ({ className, ...props }) => (
    <blockquote className={cn("aui-md-blockquote my-3 border-s-2 border-muted-foreground/40 ps-4 text-muted-foreground", className)} {...props} />
  ),
  ul: ({ className, ...props }) => (
    <ul className={cn("aui-md-ul my-3 ms-5 list-disc marker:text-muted-foreground [&>li]:mt-1", className)} {...props} />
  ),
  ol: ({ className, ...props }) => (
    <ol className={cn("aui-md-ol my-3 ms-5 list-decimal marker:text-muted-foreground [&>li]:mt-1", className)} {...props} />
  ),
  hr: ({ className, ...props }) => <hr className={cn("aui-md-hr my-3 border-border", className)} {...props} />,
  table: ({ className, ...props }) => (
    <div className="aui-md-table-wrapper my-3 overflow-x-auto">
      <table className={cn("aui-md-table w-full border-separate border-spacing-0", className)} {...props} />
    </div>
  ),
  th: ({ className, ...props }) => (
    <th className={cn("aui-md-th bg-muted px-3 py-1.5 text-start font-medium first:rounded-ss-lg last:rounded-se-lg", className)} {...props} />
  ),
  td: ({ className, ...props }) => (
    <td className={cn("aui-md-td border-s border-b border-border px-3 py-1.5 text-start last:border-e", className)} {...props} />
  ),
  tr: ({ className, ...props }) => (
    <tr className={cn("aui-md-tr m-0 border-b p-0 first:border-t", className)} {...props} />
  ),
  li: ({ className, ...props }) => <li className={cn("aui-md-li leading-relaxed", className)} {...props} />,
  strong: ({ className, ...props }) => <strong className={cn("aui-md-strong font-semibold", className)} {...props} />,
  pre: ({ className, ...props }) => (
    <pre
      className={cn(
        "aui-md-pre overflow-x-auto rounded-b-xl rounded-t-none border border-t-0 border-border/60 bg-muted/40 p-3.5 text-[0.85em] leading-relaxed",
        className,
      )}
      {...props}
    />
  ),
  code: function Code({ className, ...props }) {
    const isCodeBlock = useIsMarkdownCodeBlock();
    return (
      <code
        className={cn(!isCodeBlock && "aui-md-inline-code rounded-md bg-muted px-1.5 py-0.5 font-mono text-[0.85em]", className)}
        {...props}
      />
    );
  },
  CodeHeader,
});

const MarkdownTextImpl: FC = () => (
  <MarkdownTextPrimitive remarkPlugins={[remarkGfm]} className="aui-md" components={defaultComponents} />
);

/** Renders a text message part as GitHub-flavoured markdown, streaming-aware. */
export const MarkdownText = memo(MarkdownTextImpl);
