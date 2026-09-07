import type { Meta, StoryObj } from "@storybook/nextjs-vite";

import { PaperCard } from "@/components/paper-card";
import { papers } from "@/content/data";

const meta = {
  title: "Research/Paper card",
  component: PaperCard,
  tags: ["autodocs"],
  parameters: { nextjs: { appDirectory: true } },
  decorators: [
    (Story) => (
      <div style={{ maxWidth: "32rem", padding: "3rem" }}>
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof PaperCard>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Covariance: Story = { args: { paper: papers[0] } };
export const InteractionField: Story = { args: { paper: papers[1] } };
export const PortfolioCertificate: Story = { args: { paper: papers[2] } };
