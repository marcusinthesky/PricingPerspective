import type { Meta, StoryObj } from "@storybook/nextjs-vite";

import { VideoFeature } from "@/components/video-feature";

const meta = {
  title: "Research/Video feature",
  component: VideoFeature,
  tags: ["autodocs"],
  decorators: [
    (Story) => (
      <div style={{ padding: "3rem" }}>
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof VideoFeature>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};
